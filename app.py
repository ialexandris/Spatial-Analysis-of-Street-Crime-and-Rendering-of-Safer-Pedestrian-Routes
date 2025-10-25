from flask import Flask, request, jsonify, send_from_directory
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from time import time
import os
import traceback
import webbrowser

# -------------------------------------------------------------------
# ΡΥΘΜΙΣΕΙΣ
# -------------------------------------------------------------------
STATIC_DIR = "static"
ROADS_DIR = "roads"
ROADS_SHP = os.path.join(ROADS_DIR, "roads.shp")
ROADS_GEOJSON_4326 = os.path.join(STATIC_DIR, "roads2.geojson")

EPSG_ROADS = 27700   # OSGB36 / British National Grid
EPSG_WEB = 4326      # WGS84 (Leaflet)

print("Application loaded successfully!")

app = Flask(__name__, static_folder=STATIC_DIR)

# -------------------------------------------------------------------
# ΒΟΗΘΗΤΙΚΑ
# -------------------------------------------------------------------

def explode_multilines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Σπάει MultiLineString σε απλά LineString rows."""
    exploded = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, MultiLineString):
            for part in geom.geoms:
                r = row.copy()
                r.geometry = part
                exploded.append(r)
        else:
            exploded.append(row)
    return gpd.GeoDataFrame(exploded, crs=gdf.crs)

def parse_speed_kmh(value, default_kmh=30.0):
    try:
        if value is None:
            return float(default_kmh)
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower()
        contains_mph = 'mph' in s
        num = ''
        for ch in s:
            if ch.isdigit() or ch in ('.', ','):
                num += ('.' if ch == ',' else ch)
            elif num:
                break
        if not num:
            return float(default_kmh)
        val = float(num)
        if contains_mph:
            val *= 1.609344
        return val
    except Exception:
        return float(default_kmh)

def get_row_speed_mps(row, default_kmh=30.0, min_kmh=5.0, max_kmh=130.0):
    candidate_fields = ['risk_cost','allowed_speed','speed_kmh','maxspeed','speed','min_max']
    speed_kmh = None
    for f in candidate_fields:
        if f in row and row[f] is not None:
            speed_kmh = parse_speed_kmh(row[f], default_kmh=default_kmh)
            break
    if speed_kmh is None:
        speed_kmh = float(default_kmh)
    speed_kmh = max(min_kmh, min(max_kmh, float(speed_kmh)))
    return speed_kmh * (1000.0/3600.0)

def get_node_id(coord):
    return f"{round(coord[0], 5)}_{round(coord[1], 5)}"

def node_roads_by_intersections(roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Δημιουργεί πλήρως ‘noded’ δίκτυο μέσω unary_union και επιστρέφει segments."""
    union_geom = unary_union(list(roads_gdf.geometry.values))
    segs = []
    def collect(g):
        if isinstance(g, LineString):
            segs.append(g)
        elif isinstance(g, MultiLineString):
            for p in g.geoms:
                segs.append(p)
        elif hasattr(g, 'geoms'):
            for gg in g.geoms:
                collect(gg)
    collect(union_geom)

    segs_gdf = gpd.GeoDataFrame({'geometry': segs}, crs=roads_gdf.crs)

    # μεταφέρουμε βασικά πεδία ταχύτητας με sjoin_nearest (προαιρετικά)
    attr_cols = [c for c in ['risk_cost','allowed_speed','speed_kmh','maxspeed','speed','min_max'] if c in roads_gdf.columns]
    if attr_cols:
        joined = gpd.sjoin_nearest(segs_gdf, roads_gdf[['geometry'] + attr_cols], how='left', distance_col='join_dist').drop(columns=['join_dist'], errors='ignore')
        return joined
    return segs_gdf

# -------------------------------------------------------------------
# ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ + ΠΑΡΑΓΩΓΗ roads2.geojson (4326)
# -------------------------------------------------------------------
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

print("Loading roads...")
roads = gpd.read_file(ROADS_SHP)
if roads.crs is None or roads.crs.to_epsg() != EPSG_ROADS:
    # Αν λείπει/λάθος το .prj, το ορίζουμε ρητά
    roads = roads.set_crs(epsg=EPSG_ROADS, allow_override=True)
print(f"Loaded {len(roads)} lines (EPSG:{EPSG_ROADS}).")

# Φτιάχνουμε το GeoJSON προβολής ΜΕ ΤΟΝ ΙΔΙΟ ΜΕΤΑΣΧΗΜΑΤΙΣΜΟ
print("Exporting static/roads2.geojson (EPSG:4326) from the same pipeline...")
roads_4326 = roads.to_crs(epsg=EPSG_WEB)
roads_4326.to_file(ROADS_GEOJSON_4326, driver="GeoJSON")
print(f"Saved {ROADS_GEOJSON_4326} with {len(roads_4326)} features.")

# Χτίζουμε γράφο στο 27700
print("Building noded graph...")
t0 = time()
roads_qgis = node_roads_by_intersections(roads)
roads_qgis = explode_multilines(roads_qgis)
roads_qgis['source'] = roads_qgis.geometry.apply(lambda g: get_node_id(g.coords[0]))
roads_qgis['target'] = roads_qgis.geometry.apply(lambda g: get_node_id(g.coords[-1]))

G = nx.Graph()
nodes_xy = {}

for _, r in roads_qgis.iterrows():
    src = r['source']; tgt = r['target']; geom = r.geometry
    v = get_row_speed_mps(r)  # m/s
    w = geom.length / max(0.1, v)
    G.add_edge(src, tgt, weight=w, geometry=geom, speed_mps=v)
    nodes_xy[src] = geom.coords[0]; nodes_xy[tgt] = geom.coords[-1]

print(f"Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

# -------------------------------------------------------------------
# DIJKSTRA
# -------------------------------------------------------------------
def dijkstra_shortest_path(graph, source, target, weight_attr='weight'):
    import heapq
    dist, prev, visited = {}, {}, set()
    for n in graph.nodes: dist[n] = float('inf')
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        cur, u = heapq.heappop(heap)
        if u in visited: continue
        visited.add(u)
        if u == target: break
        for v in graph.neighbors(u):
            w = float(graph.get_edge_data(u, v).get(weight_attr, 1.0))
            alt = cur + w
            if alt < dist[v]:
                dist[v] = alt; prev[v] = u
                heapq.heappush(heap, (alt, v))
    if target not in prev and source != target:
        return None, float('inf')
    path = [target]
    while path[-1] != source: path.append(prev[path[-1]])
    path.reverse()
    return path, dist[target]

# -------------------------------------------------------------------
# SNAP σε γραμμή (στο ίδιο GDF που χτίσαμε το γράφο)
# -------------------------------------------------------------------
def snap_to_road(pt_proj, roads_active: gpd.GeoDataFrame):
    best = {'dist': float('inf'),'snapped': None,'src': None,'tgt': None,'geom': None,'along': None}
    for _, row in roads_active.iterrows():
        geom = row.geometry
        s = geom.project(pt_proj)
        p = geom.interpolate(s)
        d = pt_proj.distance(p)
        if d < best['dist']:
            best.update({'dist': d,'snapped': p,'src': row['source'],'tgt': row['target'],'geom': geom,'along': s})
    return best

def add_snap_node(graph, snapped_pt, edge_geom, src_id, tgt_id, along, nodes_dict, added_nodes, added_edges):
    """Εισάγει προσωρινό κόμβο πάνω σε ακμή και τη ‘σπάει’ σε 2."""
    temp_id = f"SNAP_{round(snapped_pt.x,5)}_{round(snapped_pt.y,5)}"
    if temp_id in graph.nodes:
        return temp_id

    # χειροκίνητο split με along
    coords = list(edge_geom.coords)
    total = edge_geom.length; acc = 0.0
    if along <= 0:
        left = None; right = edge_geom
    elif along >= total:
        left = edge_geom; right = None
    else:
        left_coords = [coords[0]]
        left = None; right = None
        for i in range(1, len(coords)):
            x0, y0 = coords[i-1]; x1, y1 = coords[i]
            seg_len = ((x1-x0)**2 + (y1-y0)**2)**0.5
            if acc + seg_len >= along:
                t = 0 if seg_len == 0 else (along - acc)/seg_len
                sx = x0 + t*(x1-x0); sy = y0 + t*(y1-y0)
                split_pt = (sx, sy)
                left_coords.append(split_pt)
                right_coords = [split_pt] + coords[i:]
                left = LineString(left_coords)
                right = LineString(right_coords)
                break
            else:
                left_coords.append((x1, y1)); acc += seg_len

    graph.add_node(temp_id)
    nodes_dict[temp_id] = (snapped_pt.x, snapped_pt.y)
    added_nodes.add(temp_id)

    base_speed = graph.edges[src_id, tgt_id].get('speed_mps', 30.0*(1000.0/3600.0)) if graph.has_edge(src_id, tgt_id) else 30.0*(1000.0/3600.0)

    if src_id in graph.nodes and left is not None and len(left.coords) >= 2:
        w_left = left.length / max(0.1, base_speed)
        graph.add_edge(src_id, temp_id, weight=w_left, geometry=left, speed_mps=base_speed)
        added_edges.add((src_id, temp_id))

    if tgt_id in graph.nodes and right is not None and len(right.coords) >= 2:
        w_right = right.length / max(0.1, base_speed)
        graph.add_edge(temp_id, tgt_id, weight=w_right, geometry=right, speed_mps=base_speed)
        added_edges.add((temp_id, tgt_id))

    return temp_id

# -------------------------------------------------------------------
# ROUTES API
# -------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/route", methods=["POST"])
def route():
    try:
        data = request.get_json(force=True)
        # Η JS στέλνει {lng, lat}. Εδώ είναι 4326.
        start_ll = Point(data['start']['lng'], data['start']['lat'])
        end_ll   = Point(data['end']['lng'],   data['end']['lat'])

        # Μετατροπή σε 27700 για snap/routing
        start_proj = gpd.GeoSeries([start_ll], crs=f"EPSG:{EPSG_WEB}").to_crs(epsg=EPSG_ROADS).iloc[0]
        end_proj   = gpd.GeoSeries([end_ll],   crs=f"EPSG:{EPSG_WEB}").to_crs(epsg=EPSG_ROADS).iloc[0]

        s = snap_to_road(start_proj, roads_qgis)
        e = snap_to_road(end_proj,   roads_qgis)

        added_nodes, added_edges = set(), set()
        src = add_snap_node(G, s['snapped'], s['geom'], s['src'], s['tgt'], s['along'], nodes_xy, added_nodes, added_edges)
        tgt = add_snap_node(G, e['snapped'], e['geom'], e['src'], e['tgt'], e['along'], nodes_xy, added_nodes, added_edges)

        path, total_w = dijkstra_shortest_path(G, src, tgt, 'weight')
        if not path:
            raise Exception("Δεν βρέθηκε διαδρομή.")

        # Συναρμολόγηση γεωμετρίας διαδρομής (27700 → 4326)
        coords = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge = G.edges[u, v]['geometry']
            ec = list(edge.coords)

            # διατήρηση σωστής φοράς
            uxy = nodes_xy[u]
            d_start = ((ec[0][0]-uxy[0])**2 + (ec[0][1]-uxy[1])**2)**0.5
            d_end   = ((ec[-1][0]-uxy[0])**2 + (ec[-1][1]-uxy[1])**2)**0.5
            if d_end < d_start:
                ec = ec[::-1]

            if i == 0: coords.extend(ec)
            else:      coords.extend(ec[1:])

        route_line = LineString(coords)
        start_conn = LineString([(start_proj.x, start_proj.y), (s['snapped'].x, s['snapped'].y)])
        end_conn   = LineString([(e['snapped'].x, e['snapped'].y), (end_proj.x, end_proj.y)])

        # σε 4326 για τον browser
        route_geo = gpd.GeoSeries([route_line], crs=EPSG_ROADS).to_crs(epsg=EPSG_WEB).iloc[0]
        start_conn_geo = gpd.GeoSeries([start_conn], crs=EPSG_ROADS).to_crs(epsg=EPSG_WEB).iloc[0]
        end_conn_geo   = gpd.GeoSeries([end_conn],   crs=EPSG_ROADS).to_crs(epsg=EPSG_WEB).iloc[0]

        out = {
            "route_coords": list(route_geo.coords),           # [ (lng,lat), ... ]
            "start_connection": list(start_conn_geo.coords),  # dashed
            "end_connection": list(end_conn_geo.coords),      # dashed
            "mid_connections": [],
            "length_m": round(route_line.length, 2),
            "total_length_m": round(route_line.length + start_conn.length + end_conn.length, 2)
        }

        # καθαρισμός προσωρινών κόμβων/ακμών
        for u, v in list(added_edges):
            if G.has_edge(u, v): G.remove_edge(u, v)
            elif G.has_edge(v, u): G.remove_edge(v, u)
        for n in list(added_nodes):
            if n in G: G.remove_node(n)
            nodes_xy.pop(n, None)

        return jsonify(out)

    except Exception as ex:
        print("Error:", ex)
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500

if __name__ == "__main__":
    print("Flask on http://127.0.0.1:5000")
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, port=5000,use_reloader=False)
