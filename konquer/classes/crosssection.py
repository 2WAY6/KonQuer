import datetime

import numpy as np
import shapefile


class CrossSection:
    def __init__(self):
        self.name = None
        self.p0 = None  # Start Point
        self.p1 = None  # End Point
        self.elmt_ids = []  # Elements, where p0 and p1 are within
        self.edge_ids = []  # Edges that are intersected
        self.intersections = []
        self.timesteps = []
        self.flows = []

    def to_numpy(self):
        return np.array([self.p0, self.p1])

    # At the moment, only the start and end point are imported
    def from_lineshape(self, geometry):
        self.p0 = (geometry.points[0][0], geometry.points[0][1])
        self.p1 = (geometry.points[-1][0], geometry.points[-1][1])

        if len(geometry.points) > 2:
            print("WARNUNG: Nur der Anfangs- und End-Punkt des "
                  "Kontrollquerschnitts wird gelesen.")

 
class CrossSectionCollection:
    """CSC"""
    def __init__(self):
        self.kqs_dict = None

    def import_kqs_from_shape(self, path_shape, field_name=None):
        """
        TEST
        """

        print("- Importiere Kontrollquerschnitte...")
        sf = shapefile.Reader(path_shape)
        fields = [s[0] for s in sf.fields if s[0] != 'DeletionFlag']

        kq_list = []
        for i in range(0, len(sf.shapes())):
            kq = CrossSection()
            kq.from_lineshape(sf.shape(i))
            kq_list.append(kq)

        if field_name is not None:
            index_station = fields.index(field_name)
            for i, r in enumerate(sf.records()):
                kq_list[i].name = r[index_station]
        else:
            for i in range(len(kq_list)):
                kq_list[i].name = str(i)

        self.kqs_dict = {kq.name: kq for kq in kq_list}

    # def write_csv(self, path_out_csv, kq_timeseries_dict):
    #     for kqid, kq_timeseries in kq_timeseries_dict.items():
    #         this_path_out_csv = path_out_csv[:-4] + str(kqid) + ".csv"
    #         stream_out = open(this_path_out_csv, 'w')
    #         for i, flow in enumerate(kq_timeseries):
    #             stream_out.write("{};{}\n".format(i, abs(sum(flow))))
    #         stream_out.close()

    def write_wel(self, path_out):
        timesteps = list(self.kqs_dict.values())[0].timesteps

        values = [[] for i in range(len(timesteps))]
        kqids = []
        for kqid, kq in self.kqs_dict.items():
            kqids.append(kqid)
            for i, flow in enumerate(kq.flows):
                values[i].append(flow)

        stream_out = open(path_out, 'w')
        stream_out.write("*WEL 01.01.2000 00:00\n")
        stream_out.write(";Datum_Zeit;{}\n".format(';'.join(map(str, kqids))))
        stream_out.write(";-;{}\n".format(';'.join(["m3/s" for i in range(len(self.kqs_dict))])))

        t0 = datetime.datetime(2000, 1, 1, 0, 0, 0)
        format = "%d.%m.%Y %H:%M"  # :%S"
        for i, seconds in enumerate(timesteps):
            ti = t0 + datetime.timedelta(seconds=float(timesteps[i]))
            ti_str = ti.strftime(format)
            stream_out.write(";{};{}\n".format(ti_str, ';'.join(map(str, values[i]))))
        stream_out.close()
