import sys
import datetime
import time

import numpy as np
import shapefile

from kontrollquerschnitt.classes import Mesh


def import_2dm_mesh(path_mesh):
    print("- Lese 2dm-Mesh...")
    nodes = []
    edges = []
    elements = []
    for line in open(path_mesh):
        if line.startswith("ND "):
            nodes.append(list(map(float, line.split()[2:])))
        elif line.startswith("E3T") or line.startswith("E4Q"):
            if line.startswith("E3T"):
                nd_indices = [int(nid) - 1 for nid in line.split()[2:5]]
            else:
                nd_indices = [int(nid) - 1 for nid in line.split()[2:6]]
            # nd_indices = [int(nid) - 1 for nid in line.split()[2:-1]]
            elements.append(nd_indices)
            c = len(nd_indices)
            for i in range(c):
                min_i = min(nd_indices[i], nd_indices[(i + 1) % c])
                max_i = max(nd_indices[i], nd_indices[(i + 1) % c])
                edges.append(f"{min_i} {max_i}")

    edges = np.array([[int(edge.split()[0]), int(edge.split()[1])] for edge in list(set(edges))])

    nodes_array = np.zeros((len(nodes), 5))
    nodes_array[:,(0,1,2)] = np.array(nodes)

    return nodes_array, edges, elements


def import_uro_mesh(path_mesh):
    print("- Lese UnRunOff-Mesh...")
    lines = [line for line in open(path_mesh).readlines() if not line.startswith('C')]

    n_bndnodes = int(lines[0])
    n_nodes = int(lines[1])
    n_elmts = int(lines[2 + n_nodes + n_bndnodes])

    i_nodes = [2, 2 + n_nodes + n_bndnodes]
    i_elmts = [2 + n_nodes + n_bndnodes + 1, 2 + n_nodes + n_bndnodes + 1 + n_elmts]

    def line_to_node(line):
        return [float(x) for x in line.split()[1:]]

    def line_to_elmt(line):
        return [int(nid) for nid in line.split()[:3]]

    nodes = [line_to_node(line) for line in lines[i_nodes[0]:i_nodes[1]]]
    nodes_array = np.zeros((len(nodes), 5))
    nodes_array[:,(0,1,2)] = np.array(nodes)

    elements = [line_to_elmt(line) for line in lines[i_elmts[0]:i_elmts[1]]]

    edges = []
    for nids in elements:
        for i in range(3):
            min_i = min(nids[i], nids[(i + 1) % 3])
            max_i = max(nids[i], nids[(i + 1) % 3])
            edges.append(f"{min_i} {max_i}")
    edges = np.array([[int(edge.split()[0]), int(edge.split()[1])] for edge in list(set(edges))])

    return nodes_array, edges, elements


def import_mesh(path_mesh):
    t0 = time.time()
    if path_mesh.endswith("2dm"):
        nodes_array, edges, elements = import_2dm_mesh(path_mesh)
    elif path_mesh.endswith("dat"):
        nodes_array, edges, elements = import_uro_mesh(path_mesh)
    else:
        sys.exit("Netzformat nicht unterstuetzt. Programmabbruch.")
    print("  - {} Knoten, {} Elemente und {} Kanten importiert.".format(len(nodes_array), len(elements), len(edges)))
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    print("- Verknuepfe Kanten und Knoten...")
    t0 = time.time()
    node_edge_link = [[] for i in range(nodes_array.shape[0])]
    for ei, edge in enumerate(edges):
        node_edge_link[edge[0]].append(ei)
        node_edge_link[edge[1]].append(ei)
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    print("- Verknuepfe Elemente und Knoten...")
    t0 = time.time()
    node_elmt_link = [[] for i in range(nodes_array.shape[0])]
    for eid, node_ids in enumerate(elements):
        for nid in node_ids:
            node_elmt_link[nid].append(eid)
    for nid, eids in enumerate(node_elmt_link):
        node_elmt_link[nid] = list(set(eids))
    print("  -> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    mesh = 

    return nodes_array, elements, node_elmt_link, edges, node_edge_link


def import_kontrollquerschnitte_from_shape(path_shape, field_id=None):
    print("- Importiere Kontrollquerschnitte...")
    sf = shapefile.Reader(path_shape)
    fields = [s[0] for s in sf.fields if s[0] != 'DeletionFlag']

    kontrollquerschnitte = []
    for i in range(0, len(sf.shapes())):
        # kontrollquerschnitte.append(np.array([(pnt[0], pnt[1]) for pnt in sf.shape(i).points]))
        p0 = (sf.shape(i).points[0][0], sf.shape(i).points[0][1])
        p1 = (sf.shape(i).points[-1][0], sf.shape(i).points[-1][1])
        kontrollquerschnitte.append(np.array([p0, p1]))

    kq_ids = []
    if field_id is not None:
        index_station = fields.index(field_id)
        for i, r in enumerate(sf.records()):
            kq_ids.append(r[index_station])
    else:
        kq_ids = [i for i in range(len(kontrollquerschnitte))]

    kqs_dict = {kq_id: kq for kq_id, kq in zip(kq_ids, kontrollquerschnitte)}

    return kqs_dict, kq_ids


def write_csv(path_out_csv, kq_timeseries_dict):
    for kqid, kq_timeseries in kq_timeseries_dict.items():
        this_path_out_csv = path_out_csv[:-4] + str(kqid) + ".csv"
        stream_out = open(this_path_out_csv, 'w')
        for i, flow in enumerate(kq_timeseries):
            stream_out.write("{};{}\n".format(i, abs(sum(flow))))
        stream_out.close()


def write_wel(path_out, kq_timeseries_dict, timesteps):
    values = [[] for i in range(len(timesteps))]
    kqids = []
    for kqid, kq_timeseries in kq_timeseries_dict.items():
        kqids.append(kqid)
        flows = [sum(flow) for flow in kq_timeseries]
        for i, flow in enumerate(flows):
            values[i].append(flow)

    stream_out = open(path_out, 'w')
    stream_out.write("*WEL 01.01.2000 00:00\n")
    stream_out.write(";Datum_Zeit;{}\n".format(';'.join(map(str, kqids))))
    stream_out.write(";-;{}\n".format(';'.join(["m3/s" for i in range(len(kq_timeseries_dict))])))

    t0 = datetime.datetime(2000, 1, 1, 0, 0, 0)
    format = "%d.%m.%Y %H:%M" #:%S"
    for i, seconds in enumerate(timesteps):
        ti = t0 + datetime.timedelta(seconds=float(timesteps[i]))
        ti_str = ti.strftime(format)
        stream_out.write(";{};{}\n".format(ti_str, ';'.join(map(str, values[i]))))
    stream_out.close()