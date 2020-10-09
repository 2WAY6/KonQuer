#!python3.7
# -*- coding: utf-8 -*-

'''

Created:    February 2020
Modified:   April 2020

@author:    Pascal Wiese

Status:     Tested and published internally

'''

import sys
import os
import argparse
import time

import numpy as np
from scipy.spatial import KDTree

from kontrollquerschnitt.inout import import_mesh, import_kontrollquerschnitte_from_shape, write_wel
from kontrollquerschnitt.functions import run_kq
from kontrollquerschnitt.plotting import plot_kqs


# TODO: improve speed (cython)
# TODO: documentation (docstrings)
# TODO: qgis implementation
# TODO: Check other TODOs in code


def main():
    # folder = r"e:\Temp\kq\tmp"
    # path_mesh = os.path.join(folder, "system.dat")
    # path_depth = None
    # path_veloc = None
    # path_erg = os.path.join(folder, "ergqh.bin")
    # path_shp = os.path.join(folder, "Kontrollquerschnitte_Selektion_out.shp")
    # field_id = "Node"
    # ts0 = 0
    # ts1 = 50000
    # modulo = 1
    # plot = False

    path_mesh, path_shp, field_id, path_depth, path_veloc, path_erg, ts0, ts1, modulo, plot, ts_plot, save, prefix = parse_args()

    if path_depth is None and path_veloc is None and path_erg is None:
        sys.exit("\nKeine Ergebnisdaten angegeben. Programmabbruch.")

    print("\nImportiere Eingangsdaten...")
    kqs_dict, kq_ids = import_kontrollquerschnitte_from_shape(path_shp, field_id=field_id)
    nodes, elements, node_elmt_link, edges, node_edge_link = import_mesh(path_mesh)

    print("\nBaue raeumliche Suchstruktur...")
    t0 = time.time()
    kdtree = KDTree(nodes[:,0:2])
    print("-> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))


    print("\nErmittle {} Kontrollquerschnitte...".format(len(kqs_dict)))
    kq_timeseries_dict, timesteps = run_kq(nodes, elements, node_elmt_link, edges, node_edge_link, kqs_dict, kq_ids,
                                           path_depth, path_veloc, path_erg, ts0, ts1, modulo, kdtree, ts_plot)

    if plot or save:
        print("\nPlotte Kontrollquerschnitte...")
        plot_kqs(kq_timeseries_dict, timesteps, plot=plot, saveplots=save, folder=os.path.dirname(path_shp), prefix=prefix)

    path_wel = path_shp[:-4] + ".wel"
    if prefix is not None:
        path_wel = os.path.join(os.path.dirname(path_shp), prefix + "_" + os.path.basename(path_shp)[:-4] + ".wel")

    print("\nSchreibe Kontrollquerschnitte als {}...".format(os.path.basename(path_wel)))
    write_wel(path_wel, kq_timeseries_dict, timesteps)


def parse_args():
    description = ("Beschreibung folgt")
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description=description, epilog="$Kontrollquerschnitte",
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mesh", metavar='netz.2dm', type=str, required=True,
                    help="Name oder Pfad des Meshs (2dm oder uro)")
    ap.add_argument("--dat", metavar='DEPTH.dat,VELOC.dat', type=str, required=False, default=None,
                    help="Name oder Pfad der depth.dat und veloc.dat mit Komma getrennt (Momentan nur ASCII-dats moeglich).")
    ap.add_argument("--erg", metavar='ergqh.bin', type=str, required=False, default=None,
                    help="Name oder Pfad der ergqh.bin")
    ap.add_argument("--shp", metavar='lines.shp', type=str, required=True,
                    help="Shape mit Kontrollquerschnitten")
    ap.add_argument("--field", metavar='id', type=str, required=False, default=None,
                    help="Feld des Shapes zur Identifikation (Text oder Zahl)")
    ap.add_argument("--ts0", metavar='0', type=float, required=False, default=-1,
                    help="Erster Zeitschritt der Auswertung in Sekunden.")
    ap.add_argument("--ts1", metavar='120', type=float, required=False, default=np.inf,
                    help="Letzter Zeitschritt der Auswertung in Sekunden.")
    ap.add_argument("--skip", metavar='1', type=int, required=False, default=1,
                    help="Nur jeden x-ten Zeitschritt auswerten.")
    ap.add_argument("--plot", metavar='false', type=str, required=False, default='false',
                    help="Ergebnisse direkt plotten.")
    ap.add_argument("--save", metavar='false', type=str, required=False, default='false',
                    help="Ergebnisse direkt als Bild speichern.")
    ap.add_argument("--prefix", metavar='hq100', type=str, required=False, default=None,
                    help="Praerix fuer die Ergebnisse.")
    ap.add_argument("--ts_plot", metavar='false', type=str, required=False, default='false',
                    help="Spezifische Durchfluesse fuer jeden Zeitschritt plotten.")
    args = vars(ap.parse_args())

    folder = '.'
    if os.path.isabs(args['mesh']):
        path_mesh = args['mesh']
    else:
        path_mesh = os.path.join(folder, args['mesh'])

    if os.path.isabs(args['shp']):
        path_shp = args['shp']
    else:
        path_shp = os.path.join(folder, args['shp'])


    if args['dat'] is not None:
        name_depth, name_veloc = [d for d in args['dat'].split(',')]
        if os.path.isabs(name_depth):
            path_depth = name_depth
        else:
            path_depth = os.path.join(folder, name_depth)
        if os.path.isabs(name_veloc):
            path_veloc = name_veloc
        else:
            path_veloc = os.path.join(folder, name_veloc)
    else:
        path_depth, path_veloc = None, None


    if args['erg'] is not None:
        if os.path.isabs(args['erg']):
            path_erg = args['erg']
        else:
            path_erg = os.path.join(folder, args['erg'])
    else:
        path_erg = None


    ts0 = args['ts0']
    ts1 = args['ts1']
    modulo = args['skip']
    field_id = args['field']
    plot = True if args['plot'].upper() in ["TRUE", "JA", "1"] else False
    save = True if args['save'].upper() in ["TRUE", "JA", "1"] else False
    ts_plot = True if args['ts_plot'].upper() in ["TRUE", "JA", "1"] else False
    prefix = args['prefix']

    return path_mesh, path_shp, field_id, path_depth, path_veloc, path_erg, ts0, ts1, modulo, plot, ts_plot, save, prefix


if __name__=='__main__':
    main()