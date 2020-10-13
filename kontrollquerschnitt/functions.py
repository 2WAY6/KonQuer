import sys
from math import sqrt
import time

import numpy as np

from .geometry import dist_2d, line_intersection, point_in_element
from .plotting import plot_ortho_flows
import cython_geometry as cy_geo


def run_kq(mesh, kqs_dict, path_dict,
           ts0, ts1, modulo, kdtree, plot=False):
    print("- Ermittle Schnittpunkte der Kontrollquerschnitte mit den Netzkanten...")
    t0 = time.time()
    calc_kq_edge_intersections(mesh, kqs_dict, kdtree)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    print("- Ermittle Element von Anfangs- und Endpunkte der Kontrollquerschnitte...")
    t0 = time.time()
    calc_kq_elmt_intersections(mesh, kqs_dict, kdtree)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    if plot:
        for kqid, kq in kqs_dict.items():
            print("\nSchnittpunkte mit Kanten von {}:".format(kqid))
            for inter in kq.intersections:
                print("{}\t{}".format(round(inter[0], 2), round(inter[1], 2)))

    print("- Ermittle Durchfluesse pro Kontrollquerschnitt je Zeitschritt...")
    t0 = time.time()
    if path_dict['depth'] is not None and path_dict['veloc'] is not None:
        calc_timeseries_dat(mesh, kdtree, kqs_dict, path_dict, ts0, ts1, 
                            modulo, plot)
    elif path_dict['erg'] is not None:
        calc_timeseries_erg(mesh, kdtree, kqs_dict, path_dict, ts0, ts1, 
                            modulo, plot)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))


def calc_timeseries_dat(mesh, kdtree, kqs_dict, path_dict, ts0, ts1, modulo, plot=False):
    flow_vectors = []
    tsi = 0
    timesteps = []
    for line_d, line_v in zip(open(path_dict['depth']), open(path_dict['veloc'])):
        try:
            depth = float(line_d)
            veloc = [float(v) for v in line_v.split()]
            flow_vectors.append((depth*veloc[0], depth*veloc[1]))

        except:
            if line_d.startswith("TS"):
                ts = float(line_d.split()[1])
                if ts > ts1:
                    break
                if ts < ts0:
                    continue
                if tsi % modulo != 0:
                    continue

                print("Bearbeite Zeitschritt {}".format(ts), end='\r')
                timesteps.append(ts)
                if flow_vectors == []:  # First timestep
                    continue
                else:
                    for kq_id in kqs_dict.keys():
                        calc_flow_for_timestep(mesh, kdtree, kqs_dict, kq_id,
                                               flow_vectors, ts, plot)

                tsi += 1
                flow_vectors = []

    # Last timestep values still have to be calculated
    for kq_id, kq in kqs_dict.items():
        calc_flow_for_timestep(mesh, kdtree, kqs_dict, kq_id, flow_vectors, ts,
                               plot)


def calc_timeseries_erg(mesh, kdtree, kqs_dict, path_dict, ts0, ts1, modulo, plot=False):
    flow_vectors = []
    timesteps = []
    n_nodes = mesh.nodes_array.shape[0]

    # 4byte
    # float Zeit
    # 4byte
    # 4byte
    # node_count * qx, qy, h (*node_count)
    # 4byte
    values_per_timestep = 1 + 3 * n_nodes + 4
    flow_vectors = np.zeros((n_nodes, 2), dtype=np.float)
    f = open(path_dict['erg'])
    tsi = 0
    while 1:
        data_ts = np.fromfile(f, dtype='float32', count=values_per_timestep)
        if data_ts.size < values_per_timestep:
            break

        if data_ts.size < values_per_timestep:
            break

        ts = data_ts[1]
        if ts > ts1:
            break
        if ts < ts0:
            continue
        if tsi % modulo != 0:
            continue

        print(f"- Bearbeite Zeitschritt {ts}", end='\r')
        timesteps.append(ts)
        values = data_ts[4:-1].reshape([n_nodes, 3])
        flow_vectors[:] = values[:, (0, 1)]
        flow_vectors[:, 0] = flow_vectors[:, 0] * values[:, 2]
        flow_vectors[:, 1] = flow_vectors[:, 1] * values[:, 2]

        for kq_id in kqs_dict.keys():
            calc_flow_for_timestep(mesh, kdtree, kqs_dict, kq_id, flow_vectors,
                                   ts, plot)
        tsi += 1


def calc_flow_for_timestep(mesh, kdtree, kqs_dict, kq_id, node_flow_vectors,
                           ts, plot=False):
    kq_flows = []
    ortho_flows = []
    kq = kqs_dict[kq_id]

    kq_dir = (kq[0, 0] - kq[1, 0], kq[0, 1] - kq[1, 1])
    # kq_ortho_dir = (-1 * kq_dir[1], kq_dir[0])
    kq_ortho_dir = (kq_dir[1], -1*kq_dir[0])
    kq_ortho_mag = sqrt(kq_ortho_dir[0] ** 2 + kq_ortho_dir[1] ** 2)

    try:
        elmt1 = mesh.elements[kq.elmt_ids[0]]
        elmt2 = mesh.elements[kq.elmt_ids[1]]

    except IndexError:
        print("IndexError:")
        print("elements[kq_elmt_ids[0]]")
        print("elements[kq_elmt_ids[1]]")
        print("kq_elmt_ids: {}".format(kq.elmt_ids))
        sys.exit("Programmabbruch")

    flows1 = np.array([node_flow_vectors[nid] for nid in elmt1])
    flows2 = np.array([node_flow_vectors[nid] for nid in elmt2])

    nodes1 = mesh.nodes_array[elmt1]
    nodes2 = mesh.nodes_array[elmt2]

    flow_start = get_flow_at_position(kq[0], nodes1, flows1)
    flow_end = get_flow_at_position(kq[1], nodes2, flows2)

    ortho_flow_start = (flow_start[0] * kq_ortho_dir[0] +
                        flow_start[1] * kq_ortho_dir[1]) / kq_ortho_mag
    ortho_flow_end = (flow_end[0] * kq_ortho_dir[0] +
                      flow_end[1] * kq_ortho_dir[1]) / kq_ortho_mag

    ortho_flows.append(ortho_flow_start)

    for i, I in enumerate(kq.intersections):
        nid1 = mesh.edges[kq.edge_ids[i]][0]
        nid2 = mesh.edges[kq.edge_ids[i]][1]
        A = mesh.nodes_array[nid1]
        B = mesh.nodes_array[nid2]
        qA = node_flow_vectors[nid1]
        qB = node_flow_vectors[nid2]

        ratio = dist_2d(A, I) / dist_2d(A, B)
        flow = (qA[0] + ratio * (qB[0] - qA[0]),
                qA[1] + ratio * (qB[1] - qA[1]))

        ortho_flow = (flow[0] * kq_ortho_dir[0] +
                      flow[1] * kq_ortho_dir[1]) / kq_ortho_mag
        ortho_flows.append(ortho_flow)

    ortho_flows.append(ortho_flow_end)

    intersections = [kq[0]] + kq.intersections + [kq[1]]

    kq_flow = 0
    for i, pnt in enumerate(intersections[:-1]):
        width = dist_2d(intersections[i], intersections[i+1])
        f1 = ortho_flows[i]
        f2 = ortho_flows[i+1]
        integral = (f1 + f2) / 2 * width
        kq_flow = integral
        kq_flows.append(kq_flow)

    if plot:
        # print(kq_flows)
        if list(set(ortho_flows)) != [0]:
            plot_ortho_flows(intersections, ortho_flows, ts,
                             round(sum(kq_flows), 3))

    return kq_flows


def get_flow_at_position(P, nodes, flows):
    if len(nodes) == 3:
        return get_flow_at_position_barycentric_weighted(P, nodes, flows)
    else:
        return get_flow_at_position_inverse_distance_weighted(P, nodes, flows)


def get_flow_at_position_inverse_distance_weighted(P, nodes, flows):
    flow = [0, 0]
    cnt = len(nodes)
    for i, N in enumerate(nodes):
        dist = dist_2d(P, N)
        if dist == 0:
            flow = flows[i]
            break

        flow[0] += 1/dist * flows[i][0]
        flow[1] += 1/dist * flows[i][1]
    return flow


# USE ONLY FOR QUADS
def get_flow_at_position_barycentric_weighted(P, nodes, flows):
    flow = [0, 0]
    weights = get_barycentric_interpolation_weights(P, nodes)

    cnt = len(nodes)
    for i, N in enumerate(nodes):
        dist = dist_2d(P, N)
        if dist == 0:
            flow = flows[i]
            break

        flow[0] += weights[i] * flows[i][0]
        flow[1] += weights[i] * flows[i][1]
    return flow


# ONLY FOR TRIS
def get_barycentric_interpolation_weights(P, nodes):
    v1, v2, v3 = nodes
    w1 = (((v2[1] - v3[1]) * (P[0] - v3[0]) + (v3[0] - v2[0]) * (P[1] - v3[1])) /
          ((v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])))
    w2 = (((v3[1] - v1[1]) * (P[0] - v3[0]) + (v1[0] - v3[0]) * (P[1] - v3[1])) /
          ((v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])))
    w3 = 1 - w1 - w2
    return [w1, w2, w3]


def calc_kq_elmt_intersections(mesh, kqs_dict, kdtree):
    for kq_id, kq in kqs_dict.items():
        kq_pts = kq.to_numpy()

        nid1 = kdtree.query(kq_pts[0])[1]
        nid2 = kdtree.query(kq_pts[1])[1]

        start_eid_candidates = mesh.node_elmt_link[nid1]
        end_eid_candidates = mesh.node_elmt_link[nid2]

        kqs_dict[kq_id].elmt_ids = []
        for eid in start_eid_candidates:
            if point_in_element(kq_pts[0], mesh.nodes[mesh.elements[eid]]):
                kqs_dict[kq_id].elmt_ids.append(eid)
                break
        for eid in end_eid_candidates:
            if point_in_element(kq_pts[1], mesh.nodes[mesh.elements[eid]]):
                kqs_dict[kq_id].elmt_ids.append(eid)
                break

        if len(kqs_dict[kq_id].elmt_ids) != 2:
            print("Start- oder Endpunkt des Kontrollquerschnitts {} liegen nicht in einem Element.".format(kq_id))
            sys.exit("Programmabbruch")


def calc_kq_edge_intersections(mesh, kqs_dict, kdtree):
    for kq_id, kq in kqs_dict.items():
        kq_pts = kq.to_numpy()
        radius = dist_2d(kq.pts[0], kq.pts[1])
        center = ((kq_pts[0, 0] + kq_pts[1, 0]) / 2,
                  (kq_pts[0, 1] + kq_pts[1, 1]) / 2)
        nd_indices = kdtree.query_ball_point([center[0], center[1]], radius)

        edge_indices = [mesh.node_edge_link[ni] for ni in nd_indices]
        edge_indices = list(set([item for sublist in edge_indices for item in sublist]))

        kq_edge_ids = []
        kq_intersections = []
        for ei in edge_indices:
            intersection = check_intersection(kq_pts, mesh.edges[ei], mesh.nodes)
            if intersection is not None:
                kq_intersections.append(intersection)
                kq_edge_ids.append(ei)

        sorter = []
        for icnt, intersection in enumerate(kq_intersections):
            dx = dist_2d(kq_pts[0], intersection)
            sorter.append([dx, icnt])

        sorter.sort(key=lambda sorter: sorter[0])

        ordered_kq_edge_ids = []
        ordered_kq_intersections = []
        for s in sorter:
            ordered_kq_edge_ids.append(kq_edge_ids[s[1]])
            ordered_kq_intersections.append(kq_intersections[s[1]])

        kqs_dict[kq_id].intersections = ordered_kq_intersections
        kqs_dict[kq_id].edge_ids = ordered_kq_edge_ids


def check_intersection(kq, edge, nodes):
    nd0 = nodes[edge[0]]
    nd1 = nodes[edge[1]]

    segment_edge = np.array([nd0[0:2], nd1[0:2]])
    segment_kq = kq

    # if segments_intersect_jit(segment_edge, segment_kq):
    if cy_geo.segments_intersect(segment_edge[0], segment_edge[1], segment_kq[0], segment_kq[1]):
        I = line_intersection(segment_edge, segment_kq)
        return I

    else:
        return None

