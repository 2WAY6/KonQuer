#!python3.8
# -*- coding: utf-8 -*-

'''

Created:    February 2020
Modified:   March 2021

@author:    Pascal Wiese

Status:     Tested and published internally

TODO:   Summenkurve optional in die wel schreiben
TODO:   Ausgabedateien am Programmausfuehrungsort und nicht am Shape-Ort
TODO:   Plot: Zeitpunkt 0 ganz links und nicht leicht versetzt
TODO:   skip -> modulo (und die anderen Parameternamen ueberdenken)
TODO:   Standardwerte der Parameter ggf. anpassen
TODO:   Meldung nur bringen, wenn len(KQ) > 2
TODO:   Anfangs- und Endpunkt-Interpolation robuster machen

'''

import sys
import os
import argparse
import time

import numpy as np
from scipy.spatial import KDTree

from classes.mesh import Mesh
from classes.crosssection import CrossSectionCollection
from analysis.flowcalc import run_kq
from visualization.plotting import plot_kqs

# from kontrollquerschnitt.inout import import_mesh
# from kontrollquerschnitt.inout import import_kqs_from_shape
# from kontrollquerschnitt.inout import write_wel
# from kontrollquerschnitt.functions import run_kq
# from kontrollquerschnitt.plotting import plot_kqs


def main():
    t0_prog = time.time()

    path_dict, field_name, params = parse_args()

    if (path_dict['depth'] is None and path_dict['veloc'] is None
            and path_dict['erg'] is None):
        sys.exit("\nKeine Ergebnisdaten angegeben. Programmabbruch.")

    print("\nImportiere Eingangsdaten...")
    cs_collection = CrossSectionCollection()
    cs_collection.import_kqs_from_shape(path_dict['shp'], field_name=field_name)

    mesh = Mesh()
    mesh.import_mesh(path_dict['mesh'])
    
    print("\nBaue raeumliche Suchstruktur...")
    t0 = time.time()
    kdtree = KDTree(mesh.nodes_array[:, 0:2])
    print("-> Nach {} Sekunden beendet.".format(round(time.time() - t0, 2)))

    print("\nErmittle {} Kontrollquerschnitte...".format(len(cs_collection.kqs_dict)))
    run_kq(mesh, cs_collection.kqs_dict, path_dict, kdtree, params)

    if params['plot'] or params['save']:
        print("\nPlotte Kontrollquerschnitte...")
        plot_kqs(cs_collection.kqs_dict, plot=params['plot'],
                 saveplots=params['save'],
                 folder=os.path.dirname(path_dict['shp']),
                 prefix=params['prefix'])

    path_wel = path_dict['shp'][:-4] + ".wel"
    if params['prefix'] is not None:
        path_wel = os.path.join(
            os.path.dirname(path_dict['shp']), params['prefix'] + "_" +
            os.path.basename(path_dict['shp'])[:-4] + ".wel")

    print("\nSchreibe Kontrollquerschnitte als {}...".format(os.path.basename(path_wel)))
    cs_collection.write_wel(path_wel)

    dt = time.time() - t0_prog
    print("\nProgramm nach {:7.2f} Minuten beendet.".format(dt/60))


def parse_args():
    description = ("Beschreibung folgt")
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser(description=description,
                                 epilog="$Kontrollquerschnitte",
                                 formatter_class=argparse.RawTextHelpFormatter)

    ap.add_argument("--mesh", metavar='netz.2dm', type=str,
                    required=True,
                    help="Name oder Pfad des Meshs (2dm oder uro)")
    ap.add_argument("--dat", metavar='DEPTH.dat,VELOC.dat', type=str,
                    required=False, default=None,
                    help="Name oder Pfad der depth.dat und veloc.dat mit Komma"
                         "getrennt (Momentan nur ASCII-dats moeglich).")
    ap.add_argument("--erg", metavar='ergqh.bin', type=str,
                    required=False, default=None,
                    help="Name oder Pfad der ergqh.bin")
    ap.add_argument("--shp", metavar='lines.shp', type=str,
                    required=True,
                    help="Shape mit Kontrollquerschnitten")
    ap.add_argument("--field", metavar='id', type=str,
                    required=False, default=None,
                    help="Feld des Shapes zur Identifikation (Text oder Zahl)")
    ap.add_argument("--ts0", metavar='0', type=float,
                    required=False, default=-1,
                    help="Erster Zeitschritt der Auswertung in Sekunden.")
    ap.add_argument("--ts1", metavar='120', type=float,
                    required=False, default=np.inf,
                    help="Letzter Zeitschritt der Auswertung in Sekunden.")
    ap.add_argument("--skip", metavar='1', type=int,
                    required=False, default=1,
                    help="Nur jeden x-ten Zeitschritt auswerten.")
    ap.add_argument("--plot", metavar='false', type=str,
                    required=False, default='false',
                    help="Ergebnisse direkt plotten.")
    ap.add_argument("--save", metavar='false', type=str,
                    required=False, default='false',
                    help="Ergebnisse direkt als Bild speichern.")
    ap.add_argument("--prefix", metavar='hq100', type=str,
                    required=False, default=None,
                    help="Praerix fuer die Ergebnisse.")
    ap.add_argument("--ts_plot", metavar='false', type=str,
                    required=False, default='false',
                    help="Spezifische Durchfluesse fuer jeden Zeitschritt"
                         "plotten.")
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

    path_dict = {'mesh': path_mesh, 'shp': path_shp, 'erg': path_erg,
                 'depth': path_depth, 'veloc': path_veloc}
    params = {'ts0': ts0,
              'ts1': ts1,
              'modulo': modulo,
              'plot': plot,
              'save': save,
              'ts_plot': ts_plot,
              'prefix': prefix}

    return path_dict, field_id, params


if __name__ == '__main__':
    main()
