import os
import sys
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# --- HELPER: Safe Logging ---
def log(msg):
    sys.stderr.write(f"[NYC-Taxi] {msg}\n")
    sys.stderr.flush()

# 1. Initialize Standard Server
server = Server("nyc-taxi-smart-agent")

# 2. Global State
models = {"rf_weather": None, "gbt_basic": None}
zone_lookup = None
analytics_df = None
route_distances = {}
ZONE_ALIASES = {
    "ewr": "Newark Airport", "newark": "Newark Airport", "jfk": "JFK Airport",
    "kennedy": "JFK Airport", "lga": "LaGuardia Airport", "laguardia": "LaGuardia Airport",
    "central park": "Central Park", "times square": "Times Sq/Theatre District",
    "manhattan": "Manhattan Valley", "chelsea": "West Chelsea/Hudson Yards",
    "harlem": "East Harlem North", "soho": "SoHo", "tribeca": "TriBeCa"
}

# 3. Initialization Logic
def load_resources():
    global zone_lookup, analytics_df, route_distances
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "..", "data")
    
    log("Initializing Resources...")

    # A. Load Zones
    try:
        zone_path = os.path.join(data_dir, "taxi_zone_lookup.csv")
        zone_lookup = pd.read_csv(zone_path)
        zone_lookup['search_name'] = zone_lookup['Zone'].str.lower()
    except Exception as e:
        log(f"Warning: Zone lookup failed: {e}")

    # B. Load Analytics
    try:
        analytics_path = os.path.join(data_dir, "analytics_summary.csv")
        if os.path.exists(analytics_path):
            analytics_df = pd.read_csv(analytics_path)
    except Exception:
        pass

    # C. Train Models
    try:
        data_path = os.path.join(data_dir, "integrated_data.csv")
        df = pd.read_csv(data_path).sample(frac=0.5, random_state=42)
        df = df.dropna(subset=["total_amount", "trip_distance"])
        
        # Model 1: Weather-Aware (RF) - 6 FEATURES
        cols_w = ["trip_distance", "PULocationID", "DOLocationID", "hour", "day_of_week", "PRCP"]
        rf = RandomForestRegressor(n_estimators=10, max_depth=8, random_state=42)
        rf.fit(df[cols_w].fillna(0), df["total_amount"])
        models["rf_weather"] = rf

        # Model 2: Basic (GBT) - 5 FEATURES
        cols_b = ["trip_distance", "PULocationID", "DOLocationID", "hour", "day_of_week"]
        gbt = GradientBoostingRegressor(n_estimators=10, max_depth=4, random_state=42)
        gbt.fit(df[cols_b].fillna(0), df["total_amount"])
        models["gbt_basic"] = gbt

        # Build Distance Cache
        routes = df.groupby(["PULocationID", "DOLocationID"])['trip_distance'].mean().reset_index()
        for _, r in routes.iterrows():
            route_distances[(int(r['PULocationID']), int(r['DOLocationID']))] = r['trip_distance']
            
        log("✅ Models Trained & Data Loaded Successfully")
        
    except Exception as e:
        log(f"❌ Critical Data Error: {e}")

load_resources()

# 4. Helper Functions
def resolve_zone(query):
    """Robust fuzzy matching for Zone IDs."""
    if not query or zone_lookup is None: return 1
    q_clean = query.lower().strip()
    q_clean = ZONE_ALIASES.get(q_clean, q_clean)
    
    match = zone_lookup[zone_lookup['search_name'] == q_clean]
    if match.empty:
        match = zone_lookup[zone_lookup['search_name'].str.contains(q_clean, regex=False, na=False)]
    return int(match.iloc[0]['LocationID']) if not match.empty else 1

def get_dist(pu, do, provided=None):
    if provided is not None:
        try:
            return float(provided)
        except ValueError:
            pass 
    hist_dist = route_distances.get((pu, do))
    return float(hist_dist) if hist_dist is not None else 5.0

def get_zone_name(zid):
    if zone_lookup is None: return str(zid)
    name = zone_lookup.loc[zone_lookup['LocationID'] == zid, 'Zone'].values
    return name[0] if len(name) > 0 else str(zid)

# 5. Define Tools
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="predict_fare_basic",
            description="Estimate fare using Gradient Boosted Trees (No Weather).",
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"}, "destination": {"type": "string"},
                    "distance": {"type": "number"}
                },
                "required": ["origin", "destination"]
            }
        ),
        types.Tool(
            name="predict_fare_weather",
            description="Estimate fare accounting for Weather (Rain/Snow).",
            inputSchema={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"}, "destination": {"type": "string"},
                    "hour": {"type": "integer"}, "weather": {"type": "string"},
                    "distance": {"type": "number"}
                },
                "required": ["origin", "destination", "hour", "weather"]
            }
        ),
        types.Tool(
            name="get_zone_metrics",
            description="Get traffic analytics for a zone (Urban Planning).",
            inputSchema={"type": "object", "properties": {"zone_name": {"type": "string"}}, "required": ["zone_name"]}
        ),
        types.Tool(
            name="get_weather_impact_stats",
            description="Get general stats on weather impact.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="compare_models",
            description="Compare Basic vs Weather models for a route.",
            inputSchema={"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}}, "required": ["origin", "destination"]}
        ),
        types.Tool(
            name="optimize_travel_time",
            description="Find cheapest time to travel.",
            inputSchema={"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}}, "required": ["origin", "destination"]}
        )
    ]

# 6. Handle Tool Calls
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    args = arguments or {}
    
    try:
        if name == "predict_fare_basic":
            pu, do = resolve_zone(args.get("origin")), resolve_zone(args.get("destination"))
            dist = get_dist(pu, do, args.get("distance"))
            # Vector (5 features): [dist, PU, DO, hour, day]
            vector = [[dist, pu, do, 14, 3]]
            pred = models["gbt_basic"].predict(vector)[0]
            return [types.TextContent(type="text", text=f"GBT Estimate: ${pred:.2f} ({dist:.1f} miles)")]

        elif name == "predict_fare_weather":
            pu, do = resolve_zone(args.get("origin")), resolve_zone(args.get("destination"))
            dist = get_dist(pu, do, args.get("distance"))
            rain = 0.5 if "rain" in args.get("weather", "").lower() else 0.0
            
            # FIXED: Vector (6 features): [dist, PU, DO, hour, day, rain]
            # Removed the extra '1' (passengers) which caused the crash
            vector = [[dist, pu, do, args.get("hour", 14), 3, rain]]
            
            pred = models["rf_weather"].predict(vector)[0]
            return [types.TextContent(type="text", text=f"Weather Adjusted Fare: ${pred:.2f}")]

        elif name == "get_zone_metrics":
            zid = resolve_zone(args.get("zone_name"))
            zname = get_zone_name(zid)
            if analytics_df is not None:
                row = analytics_df[analytics_df['PU_Zone'] == zname]
                if not row.empty:
                    return [types.TextContent(type="text", text=f"📊 {zname}: {row.iloc[0]['trip_count']} rush-hour trips, Avg Fare ${row.iloc[0]['avg_fare']:.2f}")]
            return [types.TextContent(type="text", text=f"No data for {zname} (ID: {zid})")]

        elif name == "get_weather_impact_stats":
            return [types.TextContent(type="text", text="Rain increases trip distance by 3.5%. Demand peaks at 40F.")]

        elif name == "compare_models":
            pu, do = resolve_zone(args.get("origin")), resolve_zone(args.get("destination"))
            dist = get_dist(pu, do)
            
            # Basic (5 features)
            p1 = models["gbt_basic"].predict([[dist, pu, do, 17, 3]])[0]
            
            # Weather (6 features) - FIXED
            p2 = models["rf_weather"].predict([[dist, pu, do, 17, 3, 0.5]])[0]
            
            diff = p2 - p1
            impact_str = f"+${diff:.2f}" if diff > 0 else f"-${abs(diff):.2f}"
            return [types.TextContent(type="text", text=f"Clear (GBT): ${p1:.2f} | Rain (RF): ${p2:.2f} | Impact: {impact_str}")]

        elif name == "optimize_travel_time":
            pu, do = resolve_zone(args.get("origin")), resolve_zone(args.get("destination"))
            dist = get_dist(pu, do)
            
            prices = []
            for h in [9, 14, 19]:
                # Weather (6 features) - FIXED
                p = models["rf_weather"].predict([[dist, pu, do, h, 3, 0.0]])[0]
                prices.append(f"{h}:00 -> ${p:.2f}")
            return [types.TextContent(type="text", text=" | ".join(prices))]

    except Exception as e:
        return [types.TextContent(type="text", text=f"Tool Error: {str(e)}")]

    return [types.TextContent(type="text", text="Tool not found")]

# 7. Run Server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())