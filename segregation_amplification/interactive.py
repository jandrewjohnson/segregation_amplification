"""Live, interactive view of the model: sliders change parameters while it runs, and you can edit agents with the mouse and keyboard.

Run with:  python -m segregation_amplification.interactive
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from segregation_amplification.model import SegregationMaskingModel


class InteractiveView(object):

    def __init__(self, model):

        self.model = model

        self.build_figure()
        self.connect_mpl_events()
        self.launch_timer()

        plt.show()

    def build_figure(self):
        # HACK until I make a new plotter that actually has the other things calcualte.d I don't yet calc the masks at this stage.
        a1 = self.model.masking_choice
        a2 = self.model.types_map
        a3 = np.copy(self.model.infection_status)
        a4 = self.model.immunity_efficacy
        # a4 = self.model.parameter_maps['average_reciprocal_response_from_masking']

        axis_color = 'lightgoldenrodyellow'

        self.plot_focused = False
        self.plot_lines = False

        self.fig = plt.figure()

        # Remove MPL default keyevents
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)

        if not self.plot_lines:
            self.ax_ul = self.fig.add_subplot(221)
            self.ax_ur = self.fig.add_subplot(222)
            self.ax_ll = self.fig.add_subplot(223)
            self.ax_lr = self.fig.add_subplot(224)
        else:
            self.ax_ul = self.fig.add_subplot(231)
            self.ax_ur = self.fig.add_subplot(232)
            self.ax_ll = self.fig.add_subplot(234)
            self.ax_lr = self.fig.add_subplot(235)
            self.ax_line_top = self.fig.add_subplot(233)
            self.ax_line_bottom = self.fig.add_subplot(236)

        # Adjust the subplots region to leave some space for the sliders and buttons
        self.fig.subplots_adjust(bottom=0.4)
        col_dict = {-1: "purple",
                    0: "white",
                    1: "green", }
        labels = np.array(["No mask", "Vacant", "Mask"])

        cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
        len_lab = len(labels)
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2

        self.im_ul = self.ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
        cb_ul = self.fig.colorbar(self.im_ul, ax=self.ax_ul, format=None, ticks=tickz)

        col_dict = {
            0: "white",
            1: "blue",
            2: "red"
        }
        labels = np.array(["Vacant", "Blue type", "Red type"])
        cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
        len_lab = len(labels)
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2

        self.im_ur = self.ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
        cb_ur = self.fig.colorbar(self.im_ur, ax=self.ax_ur, format=None, ticks=tickz)

        # Hack to get non-infected to be white
        a3[a3==0] = np.nan
        self.im_ll = self.ax_ll.imshow(a3, cmap='Reds_r', interpolation='nearest', vmin=1, vmax=14)
        self.im_lr = self.ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

        if self.plot_lines:
            self.ax_line_top.plot(self.model.infection_status_last_100)
            self.ax_line_top.set_ylim(0.0, 1.0)
            self.ax_line_bottom.plot(self.model.masking_choice_last_100)
            self.ax_line_bottom.set_ylim(0.0, 1.0)


        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / self.model.n_agents
        self.percent_masked_annotation = self.ax_ul.annotate('Percent masked: ' + '%.4g' % percent_masked, xy=(0.01, 0.95), xycoords='figure fraction')

        percent_infected = np.sum(np.where(self.model.infection_status >= 1, 1, 0)) / self.model.n_agents * 100.0
        self.percent_infected_annotation = self.ax_ul.annotate('Percent infected: ' + '%.4g' % percent_infected, xy=(0.51, 0.95), xycoords='figure fraction')

        self.fig.colorbar(self.im_ll, ax=self.ax_ll, shrink=.8)
        self.fig.colorbar(self.im_lr, ax=self.ax_lr, shrink=.8, extend='both')

        self.ax_ul.set_title('Masking choice')
        self.ax_ur.set_title('Political type')
        self.ax_ll.set_title('Individual r')
        self.ax_lr.set_title('Neighborhood r')

        self.ax_ul.axis('off')
        self.ax_ur.axis('off')
        self.ax_ll.axis('off')
        self.ax_lr.axis('off')

        # Turn off hover text (was breaking the window size)
        self.ax_ul.format_coord = lambda x, y: ""
        self.ax_ur.format_coord = lambda x, y: ""
        self.ax_ll.format_coord = lambda x, y: ""
        self.ax_lr.format_coord = lambda x, y: ""

        # Add Sliders
        # Define an axes area and draw a1 slider in it
        slider_properties = {}
        slider_properties['segregation_threshold'] = {'min': 0., 'max': 1., 'init': self.model.params['segregation_threshold']}
        slider_properties['neighborhood_radius'] = {'min': 1., 'max': 6., 'init': self.model.params['neighborhood_radius']}
        slider_properties['infection_probability'] = {'min': 0.0, 'max': .06, 'init': self.model.params['infection_probability']}
        slider_properties['b'] = {'min': -2., 'max': 2., 'init': self.model.params['b']}
        slider_properties['d'] = {'min': -5., 'max': 5., 'init': self.model.params['d']}
        slider_properties['s'] = {'min': -2, 'max': 2, 'init': self.model.params['s']}
        slider_properties['r'] = {'min': -5., 'max': 5., 'init': self.model.params['r']}

        if self.model.show_uncertainty_sliders:
            slider_properties['bd'] = {'min': -2., 'max': 2.}
            slider_properties['dd'] = {'min': -2., 'max': 2.}
            slider_properties['sd'] = {'min': .00001, 'max': 1.}
            slider_properties['rd'] = {'min': -5., 'max': 5.}

        i = 0
        self.sliders = {}
        for name, properties in slider_properties.items():
            slider_ax = self.fig.add_axes([.3, 0.30 - i, 0.4, 0.03], facecolor=axis_color)
            slider = Slider(slider_ax, name, properties['min'], properties['max'], valinit=properties['init'])
            slider.on_changed(self.sliders_on_changed)
            self.sliders[name] = slider
            i += .03

    def connect_mpl_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('axes_enter_event', self.on_enter_axes)
        self.fig.canvas.mpl_connect('axes_leave_event', self.on_leave_axes)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouseclick)

    def launch_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=1)
        self.timer.add_callback(self.update_function)
        self.timer.start()

        # plt.show()

    def update_function(self):

        cur_n_iterations = 1


        # ACTUALLY COMPUTE the model. This calls back to the model from the visualizer and SHOULD be the only time the model changes

        self.model.update(cur_n_iterations)

        if self.plot_lines:
            # After the model is run, update the reporting queues
            # print('np.sum(self.model.infection_status)', np.sum(self.model.infection_status))
            self.model.infection_status_last_100.insert(0, float(np.sum(np.where(self.model.infection_status>0, 1.0, 0.0))))
            self.model.infection_status_last_100.pop()

            self.model.masking_choice_last_100.insert(0, float(np.sum((self.model.masking_choice + 1) / 2)))
            self.model.masking_choice_last_100.pop()

        a1 = self.model.masking_choice
        a2 = self.model.types_map
        a3 = np.copy(self.model.infection_status)
        a4 = self.model.immunity_efficacy

        a2 = np.where(a2 > 0, a2, np.nan)
        self.im_ul.set_data(a1)
        self.im_ur.set_data(a2)
        # Hack to get non-infected to be white
        a3[a3 == 0] = np.nan
        self.im_ll.set_data(a3)
        self.im_lr.set_data(a4)

        if self.plot_lines:
            self.ax_line_top.cla()
            self.ax_line_top.plot(self.model.infection_status_last_100)
            #
            self.ax_line_bottom.cla()
            self.ax_line_bottom.plot(self.model.masking_choice_last_100)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / self.model.n_agents
        self.percent_masked_annotation.set_text('Percent masked: ' + '%.4g' % percent_masked)

        # percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / self.model.n_agents
        percent_infected = np.sum(np.where(self.model.infection_status >= 1, 1, 0)) / self.model.n_agents * 100.0
        self.percent_infected_annotation.set_text('Percent infected: ' + '%.4g' % percent_infected)

        self.fig.canvas.draw_idle()


    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(self, val):
        self.model.params['segregation_threshold'] = self.sliders['segregation_threshold'].val
        self.model.params['infection_probability'] = self.sliders['infection_probability'].val
        self.model.params['neighborhood_radius'] = self.sliders['neighborhood_radius'].val
        self.model.params['b'] = self.sliders['b'].val
        self.model.params['d'] = self.sliders['d'].val
        self.model.params['s'] = self.sliders['s'].val
        self.model.params['r'] = self.sliders['r'].val

        if self.model.show_uncertainty_sliders:
            self.model.params['bd'] = bd_slider.val
            self.model.params['dd'] = dd_slider.val
            self.model.params['sd'] = sd_slider.val
            self.model.params['rd'] = rd_slider.val

        # Because this changes the distribution of the whole population, it requires total redraw of the parameters' random values
        self.model.initialize_random_parameters()

    def on_mouseclick(self, event):

        if event.xdata is not None and self.plot_focused:

            if self.model.debug_level > 10:
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))

            r = round(event.ydata)
            c = round(event.xdata)

            val = self.model.types_map[r, c]
            print('Infected agent at ' + str(r) + ', ' + str(c) + ' who had type: ' + str(val))

            self.model.infection_status[r, c] = 1
            # self.model.parameter_maps['r'][r, c] = 3.0
            # self.model.parameter_maps['d'][r, c] = 2.0
            # self.model.types_map[r, c] = 1.0


    # Only have key commands work if you're hovering over a plot.
    def on_enter_axes(self, event):

        if len(event.inaxes.title.get_text()) > 5:
            self.plot_focused = True


    def on_leave_axes(self, event):

        if len(event.inaxes.title.get_text()) > 5:
            self.plot_focused = False


    def on_keypress(self, event):

        if self.plot_focused:

            r = round(event.ydata)
            c = round(event.xdata)

            if event.key == 'a':
                self.model.types_map[r, c] = 1.0
            if event.key == 'q':
                self.model.types_map[r, c] = 2.0
            if event.key == 's':
                self.model.parameter_maps['r'][r, c] += .5
            if event.key == 'w':
                self.model.parameter_maps['r'][r, c] -= .5
            if event.key == 'd':
                self.model.parameter_maps['d'][r, c]  += .5
            if event.key == 'e':
                self.model.parameter_maps['d'][r, c] -= .5


def main():
    parser = argparse.ArgumentParser(description='Run the model in a live, interactive window.')
    parser.add_argument('--rows', type=int, default=100)
    parser.add_argument('--cols', type=int, default=100)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    model = SegregationMaskingModel((args.rows, args.cols), seed=args.seed)
    InteractiveView(model)


if __name__ == '__main__':
    main()
