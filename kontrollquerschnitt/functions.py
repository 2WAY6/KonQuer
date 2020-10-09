import sys
from math import sqrt
import time

import numpy as np

from .geometry import dist_2d, segments_intersect_jit, line_intersection, point_in_element
from .plotting import plot_ortho_flows
# import cython_geometry as cy_geo 


def run_kq(nodes, elements, node_elmt_link, edges, node_edge_link, kqs_dict, kq_ids, path_depth, path_veloc, path_erg,
           ts0, ts1, modulo, kdtree, plot=False):
    print("- Ermittle Schnittpunkte der Kontrollquerschnitte mit den Netzkanten...")
    t0 = time.time()
    kq_edge_ids_dict, kq_intersection_dict = calc_kq_edge_intersection_dicts(nodes, edges, node_edge_link, kqs_dict, kdtree, kq_ids)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    print("- Ermittle Element von Anfangs- und Endpunkte der Kontrollquerschnitte...")
    t0 = time.time()
    kq_elmt_ids_dict = calc_kq_elmt_intersection_dicts(nodes, elements, node_elmt_link, kqs_dict, kdtree, kq_ids)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    if plot:
        for kqid, kqintersections in kq_intersection_dict.items():
            print("\nSchnittpunkte mit Kanten von {}:".format(kqid))
            for inter in kqintersections:
                print("{}\t{}".format(round(inter[0],2), round(inter[1],2)))

    print("- Ermittle Durchfluesse pro Kontrollquerschnitt je Zeitschritt...")
    t0 = time.time()
    if path_depth is not None and path_veloc is not None:
        kq_timeseries_dict, timesteps = calc_timeseries_dat(nodes, elements, edges, kdtree, kqs_dict, kq_ids,
                                                            path_depth, path_veloc, kq_edge_ids_dict,
                                                            kq_intersection_dict, kq_elmt_ids_dict, ts0, ts1, modulo,
                                                            plot)
    elif path_erg is not None:
        kq_timeseries_dict, timesteps = calc_timeseries_erg(nodes, elements, edges, kdtree, kqs_dict, kq_ids, path_erg,
                                                            kq_edge_ids_dict, kq_intersection_dict, kq_elmt_ids_dict,
                                                            ts0, ts1, modulo, plot)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    return kq_timeseries_dict, timesteps


def calc_timeseries_dat(nodes, elements, edges, kdtree, kqs_dict, kq_ids, path_depth, path_veloc, kq_edge_ids_dict,
                        kq_intersection_dict, kq_elmt_ids_dict, ts0, ts1, modulo, plot=False):
    kq_timeseries_dict = {i: [] for i in kqs_dict.keys()}
    flow_vectors = []
    tsi = 0
    timesteps = []
    for line_d, line_v in zip(open(path_depth), open(path_veloc)):
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
                if flow_vectors == []: # First timestep
                    continue
                else:
                    for kq_id, kq in kqs_dict.items():
                        kq_flow = calc_flow_for_timestep(nodes, elements, edges, kdtree, kq, kq_edge_ids_dict[kq_id],
                                                         kq_intersection_dict[kq_id], kq_elmt_ids_dict[kq_id],
                                                         flow_vectors, ts, plot)
                        kq_timeseries_dict[kq_id].append(kq_flow)

                tsi += 1
                flow_vectors = []

    # Last timestep values still have to be calculated
    for kq_id, kq in kqs_dict.items():
        kq_flow = calc_flow_for_timestep(nodes, elements, edges, kdtree, kq, kq_edge_ids_dict[kq_id],
                                         kq_intersection_dict[kq_id], kq_elmt_ids_dict[kq_id], flow_vectors, ts)
        kq_timeseries_dict[kq_id].append(kq_flow)

    return kq_timeseries_dict, timesteps


def calc_timeseries_erg(nodes, elements, edges, kdtree, kqs_dict, kq_ids, path_erg, kq_edge_ids_dict,
                        kq_intersection_dict, kq_elmt_ids_dict, ts0, ts1, modulo, plot=False):
    kq_timeseries_dict = {i: [] for i in kqs_dict.keys()}
    flow_vectors = []
    timesteps = []
    n_nodes = nodes.shape[0]

    # 4byte
    # float Zeit
    # 4byte
    # 4byte
    # node_count * qx, qy, h (*node_count)
    # 4byte
    values_per_timestep = 1 + 3 * n_nodes + 4
    flow_vectors = np.zeros((n_nodes, 2), dtype=np.float)
    f = open(path_erg)
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
        flow_vectors[:] = values[:,(0,1)]
        flow_vectors[:,0] = flow_vectors[:,0] * values[:,2]
        flow_vectors[:,1] = flow_vectors[:,1] * values[:,2]

        for kq_id, kq in kqs_dict.items():
            kq_flow = calc_flow_for_timestep(nodes, elements, edges, kdtree, kq, kq_edge_ids_dict[kq_id],
                                             kq_intersection_dict[kq_id], kq_elmt_ids_dict[kq_id], flow_vectors, ts,
                                             plot)
            kq_timeseries_dict[kq_id].append(kq_flow)
        tsi += 1

    return kq_timeseries_dict, timesteps


def calc_flow_for_timestep(nodes, elements, edges, kdtree, kq, kq_edge_ids, kq_intersections, kq_elmt_ids,
                           node_flow_vectors, ts, plot=False):
    kq_flows = []
    ortho_flows = []

    use_old_version = False

    kq_dir = (kq[0, 0] - kq[1, 0], kq[0, 1] - kq[1, 1])
    # kq_ortho_dir = (-1 * kq_dir[1], kq_dir[0])
    kq_ortho_dir = (kq_dir[1], -1*kq_dir[0])
    kq_ortho_mag = sqrt(kq_ortho_dir[0] ** 2 + kq_ortho_dir[1] ** 2)

    if use_old_version == True:
        # version using nearest node version
        i_start = kdtree.query(kq[0])[1]
        i_end = kdtree.query(kq[1])[1]
        nd_start = nodes[i_start]
        nd_end = nodes[i_end]

        flow = node_flow_vectors[i_start]
        ortho_flow_start = (flow[0] * kq_ortho_dir[0] + flow[1] * kq_ortho_dir[1]) / kq_ortho_mag
        flow = node_flow_vectors[i_end]
        ortho_flow_end = (flow[0] * kq_ortho_dir[0] + flow[1] * kq_ortho_dir[1]) / kq_ortho_mag

    # version using start and end point of kq
    else:
        try:
            elmt1 = elements[kq_elmt_ids[0]]
            elmt2 = elements[kq_elmt_ids[1]]
        except IndexError:
            print("IndexError:")
            print("elements[kq_elmt_ids[0]]")
            print("elements[kq_elmt_ids[1]]")
            print("kq_elmt_ids: {}".format(kq_elmt_ids))
            sys.exit("Programmabbruch")

        flows1 = np.array([node_flow_vectors[nid] for nid in elmt1])
        flows2 = np.array([node_flow_vectors[nid] for nid in elmt2])

        nodes1 = nodes[elmt1]
        nodes2 = nodes[elmt2]

        flow_start = get_flow_at_position(kq[0], nodes1, flows1)
        flow_end = get_flow_at_position(kq[1], nodes2, flows2)

        ortho_flow_start = (flow_start[0] * kq_ortho_dir[0] + flow_start[1] * kq_ortho_dir[1]) / kq_ortho_mag
        ortho_flow_end = (flow_end[0] * kq_ortho_dir[0] + flow_end[1] * kq_ortho_dir[1]) / kq_ortho_mag

    ortho_flows.append(ortho_flow_start)

    for i, I in enumerate(kq_intersections):
        nid1 = edges[kq_edge_ids[i]][0]
        nid2 = edges[kq_edge_ids[i]][1]
        A = nodes[nid1]
        B = nodes[nid2]
        qA = node_flow_vectors[nid1]
        qB = node_flow_vectors[nid2]

        ratio = dist_2d(A, I) / dist_2d(A, B)
        flow = (qA[0] + ratio * (qB[0] - qA[0]), qA[1] + ratio * (qB[1] - qA[1]))
        flow_mag = sqrt(flow[0]**2 + flow[1]**2)

        ortho_flow = (flow[0] * kq_ortho_dir[0] + flow[1] * kq_ortho_dir[1]) / kq_ortho_mag
        ortho_flows.append(ortho_flow)

    ortho_flows.append(ortho_flow_end)

    if use_old_version:
        intersections = [nd_start] + kq_intersections + [nd_end]
    else:
        intersections = [kq[0]] + kq_intersections + [kq[1]]


    kq_flow = 0
    for i, pnt in enumerate(intersections[:-1]):
        width = dist_2d(intersections[i], intersections[i+1])
        f1 = ortho_flows[i]
        f2 = ortho_flows[i+1]
        integral = (f1 + f2) / 2 * width
        kq_flow = integral

        # print(f"- {i}")
        # print(f"  - flow 1: {f1} m2/s")
        # print(f"  - flow 2: {f2} m2/s")
        # print(f"  - width:  {width} m")
        # print(f"  - integral:   {integral} m3/s")

        kq_flows.append(kq_flow)

    if plot:
        # print(kq_flows)
        if list(set(ortho_flows)) != [0]:
            plot_ortho_flows(intersections, ortho_flows, ts, round(sum(kq_flows), 3))

    #print("- {} = Summe von {}".format(sum(kq_flows), kq_flows))
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



def calc_kq_elmt_intersection_dicts(nodes, elements, node_elmt_link, kqs_dict, kdtree, kq_ids):
    kq_elmt_ids_dict = {}

    for kq_id, kq in kqs_dict.items():
        kq_elmt_ids_dict[kq_id] = []

        nid1 = kdtree.query(kq[0])[1]
        nid2 = kdtree.query(kq[1])[1]

        start_eid_candidates = node_elmt_link[nid1]
        end_eid_candidates = node_elmt_link[nid2]

        for eid in start_eid_candidates:
            if point_in_element(kq[0], nodes[elements[eid]]):
                kq_elmt_ids_dict[kq_id].append(eid)
                break
        for eid in end_eid_candidates:
            if point_in_element(kq[1], nodes[elements[eid]]):
                kq_elmt_ids_dict[kq_id].append(eid)
                break
        a = 1

        if len(kq_elmt_ids_dict[kq_id]) != 2:
            print("Start- oder Endpunkt des Kontrollquerschnitts {} liegen nicht in einem Element.".format(kq_id))
            sys.exit("Programmabbruch")

    return kq_elmt_ids_dict


def calc_kq_edge_intersection_dicts(nodes, edges, node_edge_link, kqs_dict, kdtree, kq_ids):
    kq_intersection_dict = {}
    kq_edge_ids_dict = {}

    for kq_id, kq in kqs_dict.items():
        radius = dist_2d(kq[0], kq[1])
        center = ((kq[0, 0] + kq[1, 0]) / 2, (kq[0, 1] + kq[1, 1]) / 2)
        nd_indices = kdtree.query_ball_point([center[0], center[1]], radius)

        edge_indices = [node_edge_link[ni] for ni in nd_indices]
        edge_indices = list(set([item for sublist in edge_indices for item in sublist]))

        kq_edge_ids = []
        kq_intersections = []
        for ei in edge_indices:
            I = check_intersection(kq, edges[ei], nodes)
            if I is not None:
                # print("")
                # print("Schnittpunkt: {}\t{}".format(round(I[0],2), round(I[1], 2)))
                # node1 = nodes[edges[ei][0]]
                # node2 = nodes[edges[ei][1]]
                # print("Knoten 1 von Kante: {}\t{}".format(round(node1[0],2), round(node1[1], 2)))
                # print("Knoten 2 von Kante: {}\t{}".format(round(node2[0],2), round(node2[1], 2)))
                kq_intersections.append(I)
                kq_edge_ids.append(ei)

        sorter = []
        for icnt, I in enumerate(kq_intersections):
            dx = dist_2d(kq[0], I)
            sorter.append([dx, icnt])

        sorter.sort(key=lambda sorter: sorter[0])

        ordered_kq_edge_ids = []
        ordered_kq_intersections = []
        for s in sorter:
            ordered_kq_edge_ids.append(kq_edge_ids[s[1]])
            ordered_kq_intersections.append(kq_intersections[s[1]])

        kq_intersection_dict[kq_id] = ordered_kq_intersections
        kq_edge_ids_dict[kq_id] = ordered_kq_edge_ids

    return kq_edge_ids_dict, kq_intersection_dict


def check_intersection(kq, edge, nodes):
    nd0 = nodes[edge[0]]
    nd1 = nodes[edge[1]]

    segment_edge = np.array([nd0[0:2], nd1[0:2]])
    segment_kq = kq

    if segments_intersect_jit(segment_edge, segment_kq):
        I = line_intersection(segment_edge, segment_kq)
        return I

    else:
        return None

