import os
from math import sqrt

import matplotlib.pyplot as plt


def plot_ortho_flows(intersections, ortho_flows, ts, summe, divisor=0.1):

    print("\nPlot Values:")
    for xy, fl in zip(intersections, ortho_flows):
        print("{}\t{}\t{}".format(round(xy[0], 2), round(xy[1], 2),
                                  round(fl, 3)))

    i_xs = [inter[0] for inter in intersections]
    i_ys = [inter[1] for inter in intersections]

    kq_dir = (intersections[-1][0] - intersections[0][0],
              intersections[-1][1] - intersections[0][1])
    kq_ortho_dir = (-1 * kq_dir[1], kq_dir[0])
    kq_ortho_dir_mag = sqrt(kq_ortho_dir[0]**2 + kq_ortho_dir[1]**2)
    kq_ortho_dir_norm = (kq_ortho_dir[0] / kq_ortho_dir_mag,
                         kq_ortho_dir[1] / kq_ortho_dir_mag)

    fig, ax = plt.subplots()
    ax.plot(i_xs, i_ys, 'bo-')

    e_x = []
    e_y = []
    for i, inter in enumerate(intersections):
        x0 = inter[0]
        y0 = inter[1]
        flow = ortho_flows[i]

        x1 = x0 + flow * kq_ortho_dir_norm[0] / divisor
        y1 = y0 + flow * kq_ortho_dir_norm[1] / divisor
        e_x.append(x1)
        e_y.append(y1)
        ax.plot([x0, x1], [y0, y1], 'r-')

        ax.text(x1, y1, round(flow, 3))

    ax.plot(e_x, e_y, 'r-')
    ax.set_title(f"Timestep {ts}\n{summe} m3/s")
    ax.axis('equal')
    plt.show()

    plt.close()


def plot_kqs(kq_timeseries_dict, timesteps, plot=True, saveplots=False,
             folder='', prefix=None):  # TODO: Make plot saveable
    timesteps_hours = [ts/3600 for ts in timesteps]

    for kqid, kq_timeseries in kq_timeseries_dict.items():
        flows = [sum(flow) for flow in kq_timeseries]
        vols = [0]
        for i, flow in enumerate(flows[:-1]):
            vol = (timesteps[i+1] - timesteps[i]) * \
                  (flows[i] + flows[i+1]) / 2  # Trapezintegration
            vols.append(vol + vols[-1])

        max_flow = 0
        max_i = 0

        for i, f in enumerate(flows):
            if f > max_flow:
                max_i = i
                max_flow = f
        max_ts = timesteps_hours[max_i]

        fig, ax1 = plt.subplots()
        color = 'blue'
        ax1.plot(timesteps_hours, flows, color=color)
        ax1.text(max_ts, max_flow, round(max_flow, 2), color='blue')
        ax1.set_title("Kontrollquerschnitt: {}".format(kqid))
        ax1.set_xlabel("Zeitschritt [h]")
        ax1.set_ylabel("Durchfluss [m³/s]", color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()
        color = 'red'
        ax2.plot(timesteps_hours, vols, color=color)
        ax2.set_ylabel("Volumen [m³]", color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()

        if saveplots:
            if prefix is None:
                pfad = os.path.join(folder, str(kqid)) + ".png"
            else:
                pfad = os.path.join(folder, prefix + "_" + str(kqid)) + ".png"
            print("- Speichere Plot unter {}".format(pfad))
            plt.savefig(pfad)

        if plot:
            plt.show()
