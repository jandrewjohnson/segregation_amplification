import numpy as np
import scipy as sp
from scipy import stats
import hazelbean as hb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib import colors

import tribal_masking_functions
import tribal_masking_computational_core
import tribal_masking_tasks

matplotlib.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer or Qt5Agg



def plot_combined_game_interactive_full_resolve(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init):


    axis_color = 'lightgoldenrodyellow'

    fig = plt.figure()
    ax_ul = fig.add_subplot(221)
    ax_ur = fig.add_subplot(222)
    ax_ll = fig.add_subplot(223)
    ax_lr = fig.add_subplot(224)

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.4)
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

    im_ul = ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
    cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)

    col_dict = {
        0: "white",
        1: "blue",
        2: "red"
    }
    labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
    cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=None, ticks=tickz)

    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

    # im_ul = ax_ul.imshow(a1, cmap='PRGn', interpolation='nearest', vmin=-1.5, vmax=1.5)
    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    # im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    # im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
    #
    percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
    percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')

    # diff = norm_bins[1:] - norm_bins[:-1]
    # tickz = norm_bins[:-1] + diff / 2
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, ticks=[-1, 1], shrink=.8 )
    # cb_ul.ax.set_yticklabels(['no mask', 'mask'])

    # fig.colorbar(im_ur, ax=ax_ur, shrink=.8)

    fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
    fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')

    ax_ul.set_title('Masking choice')
    ax_ur.set_title('Political type')
    ax_ll.set_title('Neighborhood similarity')
    ax_lr.set_title('Neighborhood r')

    ax_ul.axis('off')
    ax_ur.axis('off')
    ax_ll.axis('off')
    ax_lr.axis('off')

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        a1, a2, a3, a4 = tribal_masking_functions.get_masking_choice_full_resolve(spatial_shape, proportion_filled, game_type, threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val)
        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    # Turn off hover text (was breaking the window size)
    ax_ul.format_coord = lambda x, y: ""
    ax_ur.format_coord = lambda x, y: ""
    ax_ll.format_coord = lambda x, y: ""
    ax_lr.format_coord = lambda x, y: ""

    # Add Sliders
    # Define an axes area and draw a1 slider in it
    threshold_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
    threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=threshold_init)
    threshold_slider.on_changed(sliders_on_changed)

    neighborhood_radius_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
    neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=neighborhood_radius_init)
    neighborhood_radius_slider.on_changed(sliders_on_changed)

    type_bias_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
    type_bias_slider = Slider(type_bias_slider_ax, 'type_bias', -2., 2., valinit=type_bias_init)
    type_bias_slider.on_changed(sliders_on_changed)

    d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
    d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=d_init)
    d_slider.on_changed(sliders_on_changed)

    dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
    dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=dd_init)
    dd_slider.on_changed(sliders_on_changed)

    s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
    s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=s_init)
    s_slider.on_changed(sliders_on_changed)

    sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
    sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=sd_init)
    sd_slider.on_changed(sliders_on_changed)

    r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
    r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=r_init)
    r_slider.on_changed(sliders_on_changed)

    rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
    rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=rd_init)
    rd_slider.on_changed(sliders_on_changed)

    plt.show()



def plot_combined_game_interactive_time_variant(a1, a2, a3, a4, spatial_shape, proportion_filled, game_type, n_agents, threshold_init, neighborhood_radius_init, type_bias_init, d_init, dd_init, s_init, sd_init, r_init, rd_init):
    # Define an action for modifying the line when any slider's value changes


    def sliders_on_changed_full_resolve(val):
        a1, a2, a3, a4 = tribal_masking_functions.get_masking_choice_time_variant(spatial_shape, proportion_filled, game_type, threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val)
        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()



    def sliders_on_changed_just_masking_updates(val):
        # global a1
        # START HERE, i made this code real ugly here because i couldn't get it to use an EXISTING solution rather than just reassigning randomly or keeping the prior.
        params = {'d': [-10, 10, d_slider.val, dd_slider.val], 's': [0, 1, s_slider.val, sd_slider.val], 'r': [-10, 10, r_slider.val, rd_slider.val]}
        # params_init = {'d': [-10, 10, -.25, .0000001], 's': [0, 10, 1.0, .0000001], 'r': [-10, 10, 3.0, .0000001]}
        # d_init = params_init['d'][2]
        # dd_init = params_init['d'][3]
        # s_init = params_init['s'][2]
        # sd_init = params_init['s'][3]
        # r_init = params_init['r'][2]
        # rd_init = params_init['r'][3]

        row_ids = np.arange(spatial_shape[0])
        col_ids = np.arange(spatial_shape[1])

        # Number of agents is a function then of the size of the space and the proportion filled, rounded down and inted.
        n_agents = int(np.floor(spatial_shape[0] * spatial_shape[1] * proportion_filled))
        n_cells = spatial_shape[0] * spatial_shape[1]

        # a = sp.stats.truncnorm.rvs((params['d'][0] - params['d'][2]) / params['d'][3], (params['d'][1] - params['d'][2]) / params['d'][3], loc=params['d'][2], scale=params['d'][3], size=n_cells)
        # a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        # d_paramsial = a.reshape(spatial_shape).astype(np.float32)
        # d = np.copy(d_paramsial)
        #
        # a = sp.stats.truncnorm.rvs((params['s'][0] - params['s'][2]) / params['s'][3], (params['s'][1] - params['s'][2]) / params['s'][3], loc=params['s'][2], scale=params['s'][3], size=n_cells)
        # a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        # s_paramsial = a.reshape(spatial_shape).astype(np.float32)
        # s = np.copy(s_paramsial)
        #
        # a = sp.stats.truncnorm.rvs((params['r'][0] - params['r'][2]) / params['r'][3], (params['r'][1] - params['r'][2]) / params['r'][3], loc=params['r'][2], scale=params['r'][3], size=n_cells)
        # a = np.where(agent_ids_map.flatten() > 0, a, 0.)
        # a = nip.where(types_map.flatten() == 2, a * type_bias_slider.val, a)
        # r_intial = a.reshape(spatial_shape).astype(np.float32)
        # r = np.copy(r_initial)

        agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r = \
            tribal_masking_functions.initialize_agents(spatial_shape, proportion_filled, type_bias_slider.val, params)


        a1, a2, a3, a4 = tribal_masking_functions.update_masking_choice_without_reposition(a1_gl, a2_gl, a3_gl, a4_gl, spatial_shape, proportion_filled, game_type, threshold_slider.val, neighborhood_radius_slider.val, type_bias_slider.val, d_slider.val, dd_slider.val, s_slider.val, sd_slider.val, r_slider.val, rd_slider.val,
                                                                                           agent_ids, agent_locations, unoccupied_locations, agent_types, agent_ids_map, types_map, d, s, r)
        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    axis_color = 'lightgoldenrodyellow'

    fig = plt.figure()
    ax_ul = fig.add_subplot(221)
    ax_ur = fig.add_subplot(222)
    ax_ll = fig.add_subplot(223)
    ax_lr = fig.add_subplot(224)

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.4)
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

    im_ul = ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
    cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)

    col_dict = {
        0: "white",
        1: "blue",
        2: "red"
    }
    labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
    cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=None, ticks=tickz)

    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

    # im_ul = ax_ul.imshow(a1, cmap='PRGn', interpolation='nearest', vmin=-1.5, vmax=1.5)
    # im_ur = ax_ur.imshow(a2, cmap='Set1_r', interpolation='nearest', vmin=-7, vmax=3)
    # im_ll = ax_ll.imshow(a3, cmap='YlGn', interpolation='nearest', vmin=0, vmax=1)
    # im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
    #
    percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / n_agents
    percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')

    # diff = norm_bins[1:] - norm_bins[:-1]
    # tickz = norm_bins[:-1] + diff / 2
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)
    # cb_ul = fig.colorbar(im_ul, ax=ax_ul, ticks=[-1, 1], shrink=.8 )
    # cb_ul.ax.set_yticklabels(['no mask', 'mask'])

    # fig.colorbar(im_ur, ax=ax_ur, shrink=.8)

    fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
    fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')

    ax_ul.set_title('Masking choice')
    ax_ur.set_title('Political type')
    ax_ll.set_title('Neighborhood similarity')
    ax_lr.set_title('Neighborhood r')

    ax_ul.axis('off')
    ax_ur.axis('off')
    ax_ll.axis('off')
    ax_lr.axis('off')


    # Turn off hover text (was breaking the window size)
    ax_ul.format_coord = lambda x, y: ""
    ax_ur.format_coord = lambda x, y: ""
    ax_ll.format_coord = lambda x, y: ""
    ax_lr.format_coord = lambda x, y: ""

    # Add Sliders
    # Define an axes area and draw a slider in it
    threshold_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
    threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=threshold_init)
    threshold_slider.on_changed(sliders_on_changed_full_resolve)


    global a1_gl, a2_gl, a3_gl, a4_gl
    a1_gl = a1
    a2_gl = a2
    a3_gl = a3
    a4_gl = a4
    neighborhood_radius_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
    neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=neighborhood_radius_init)
    neighborhood_radius_slider.on_changed(sliders_on_changed_just_masking_updates)

    type_bias_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
    type_bias_slider = Slider(type_bias_slider_ax, 'type_bias', -2., 2., valinit=type_bias_init)
    type_bias_slider.on_changed(sliders_on_changed_just_masking_updates)

    d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
    d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=d_init)
    d_slider.on_changed(sliders_on_changed_just_masking_updates)

    dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
    dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=dd_init)
    dd_slider.on_changed(sliders_on_changed_just_masking_updates)

    s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
    s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=s_init)
    s_slider.on_changed(sliders_on_changed_just_masking_updates)

    sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
    sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=sd_init)
    sd_slider.on_changed(sliders_on_changed_just_masking_updates)

    r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
    r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=r_init)
    r_slider.on_changed(sliders_on_changed_just_masking_updates)

    rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
    rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=rd_init)
    rd_slider.on_changed(sliders_on_changed_just_masking_updates)

    plt.show()


def plot_combined_game_interactive_sliders(model):
    # TEMPORARY HACK until I make a new plotter that actually has the other things calcualte.d I don't yet calc the masks at this stage.
    a1 = model.masking_choice
    a2 = model.types_map
    a3 = model.parameter_maps['r']
    a4 = model.parameter_maps['average_reciprocal_response_from_masking']

    axis_color = 'lightgoldenrodyellow'

    fig = plt.figure()

    # Remove MPL default keyevents
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    ax_ul = fig.add_subplot(221)
    ax_ur = fig.add_subplot(222)
    ax_ll = fig.add_subplot(223)
    ax_lr = fig.add_subplot(224)

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.4)
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

    im_ul = ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
    cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)

    col_dict = {
        0: "white",
        1: "blue",
        2: "red"
    }
    labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
    cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=None, ticks=tickz)

    im_ll = ax_ll.imshow(a3, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)
    im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

    percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / model.n_agents
    percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')

    fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
    fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')

    ax_ul.set_title('Masking choice')
    ax_ur.set_title('Political type')
    ax_ll.set_title('Individual r')
    ax_lr.set_title('Neighborhood r')

    ax_ul.axis('off')
    ax_ur.axis('off')
    ax_ll.axis('off')
    ax_lr.axis('off')

    def update_function():

        cur_n_iterations = 1
        tribal_masking_computational_core.spatial_segregation_and_externality(model, cur_n_iterations)

        a1 = model.masking_choice
        a2 = model.types_map
        a3 = model.parameter_maps['r']
        a4 = model.parameter_maps['average_reciprocal_response_from_masking']

        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / model.n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        model.params['segregation_threshold'] = threshold_slider.val
        model.params['neighborhood_radius'] = neighborhood_radius_slider.val
        model.params['b'] = b_slider.val
        if model.show_uncertainty_sliders:
            model.params['bd'] = bd_slider.val
        model.params['d'] = d_slider.val
        if model.show_uncertainty_sliders:
            model.params['dd'] = dd_slider.val
        model.params['s'] = s_slider.val
        if model.show_uncertainty_sliders:
            model.params['sd'] = sd_slider.val
        model.params['r'] = r_slider.val
        if model.show_uncertainty_sliders:
            model.params['rd'] = rd_slider.val

        # Because this changes the distribution of the whole population, it requires total redraw of the parameters' random values
        model.initialize_random_parameters()

    def on_mouseclick(event):
        if event.xdata is not None:

            if model.debug_level > 10:
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))

            r = round(event.ydata)
            c = round(event.xdata)

            val = model.types_map[r, c]
            print('Agent at ' + str(r) + ', ' + str(c) + ' had type: ' + str(val))

            # model.parameter_maps['r'][r, c] = 3.0
            # model.parameter_maps['d'][r, c] = 2.0
            # model.types_map[r, c] = 1.0

    fig.canvas.mpl_connect('button_press_event', on_mouseclick)

    def on_keypress(event):
        print('press', event.key)
        r = round(event.ydata)
        c = round(event.xdata)

        if event.key == 'a':
            model.types_map[r, c] = 1.0
        if event.key == 'q':
            model.types_map[r, c] = 2.0
        if event.key == 's':
            model.parameter_maps['r'][r, c] += .5
        if event.key == 'w':
            model.parameter_maps['r'][r, c] -= .5
        if event.key == 'd':
            model.parameter_maps['d'][r, c]  += .5
        if event.key == 'e':
            model.parameter_maps['d'][r, c] -= .5

    fig.canvas.mpl_connect('key_press_event', on_keypress)

    timer = fig.canvas.new_timer(interval=10)
    timer.add_callback(update_function)
    timer.start()

    # Turn off hover text (was breaking the window size)
    ax_ul.format_coord = lambda x, y: ""
    ax_ur.format_coord = lambda x, y: ""
    ax_ll.format_coord = lambda x, y: ""
    ax_lr.format_coord = lambda x, y: ""

    # Add Sliders
    # Define an axes area and draw a1 slider in it
    threshold_slider_ax = fig.add_axes([.3, 0.30, 0.4, 0.03], facecolor=axis_color)
    threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=model.params['segregation_threshold'])
    threshold_slider.on_changed(sliders_on_changed)

    neighborhood_radius_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
    neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=model.params['neighborhood_radius'])
    neighborhood_radius_slider.on_changed(sliders_on_changed)

    b_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
    b_slider = Slider(b_slider_ax, 'b', -2., 2., valinit=model.params['b'])
    b_slider.on_changed(sliders_on_changed)

    if model.show_uncertainty_sliders:
        bd_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
        bd_slider = Slider(bd_slider_ax, 'bd', -2., 2., valinit=model.params['bd'])
        bd_slider.on_changed(sliders_on_changed)

    d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
    d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=model.params['d'])
    d_slider.on_changed(sliders_on_changed)

    if model.show_uncertainty_sliders:
        dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
        dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=model.params['dd'])
        dd_slider.on_changed(sliders_on_changed)

    s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
    s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=model.params['s'])
    s_slider.on_changed(sliders_on_changed)

    if model.show_uncertainty_sliders:
        sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
        sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=model.params['sd'])
        sd_slider.on_changed(sliders_on_changed)

    r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
    r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=model.params['r'])
    r_slider.on_changed(sliders_on_changed)

    if model.show_uncertainty_sliders:
        rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
        rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=model.params['rd'])
        rd_slider.on_changed(sliders_on_changed)

    plt.show()


def plot_combined_game_interactive_infections(model):
    # TEMPORARY HACK until I make a new plotter that actually has the other things calcualte.d I don't yet calc the masks at this stage.
    a1 = model.masking_choice
    a2 = model.types_map
    a3 = model.infection_status
    a4 = model.parameter_maps['average_reciprocal_response_from_masking']

    axis_color = 'lightgoldenrodyellow'

    plot_focused = False

    fig = plt.figure()

    # Remove MPL default keyevents
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    ax_ul = fig.add_subplot(221)
    ax_ur = fig.add_subplot(222)
    ax_ll = fig.add_subplot(223)
    ax_lr = fig.add_subplot(224)

    # Adjust the subplots region to leave some space for the sliders and buttons
    fig.subplots_adjust(bottom=0.4)
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

    im_ul = ax_ul.imshow(a1, cmap=cm, norm=norm, interpolation='nearest')
    cb_ul = fig.colorbar(im_ul, ax=ax_ul, format=None, ticks=tickz)

    col_dict = {
        0: "white",
        1: "blue",
        2: "red"
    }
    labels = np.array(["Vacant", "Tribe 1", "Tribe 2"])
    cm = matplotlib.colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
    len_lab = len(labels)
    norm_bins = np.sort([*col_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2

    im_ur = ax_ur.imshow(a2, cmap=cm, norm=norm, interpolation='nearest')
    cb_ur = fig.colorbar(im_ur, ax=ax_ur, format=None, ticks=tickz)

    # Hack to get non-infected to be white
    a3[a3==0] = np.nan
    im_ll = ax_ll.imshow(a3, cmap='Reds_r', interpolation='nearest', vmin=1, vmax=14)
    im_lr = ax_lr.imshow(a4, cmap='BrBG', interpolation='nearest', vmin=-2, vmax=2)

    percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / model.n_agents
    percent_masked_annotation = ax_ul.annotate('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)), xy=(0.01, 0.95), xycoords='figure fraction')

    fig.colorbar(im_ll, ax=ax_ll, shrink=.8)
    fig.colorbar(im_lr, ax=ax_lr, shrink=.8, extend='both')

    ax_ul.set_title('Masking choice')
    ax_ur.set_title('Political type')
    ax_ll.set_title('Individual r')
    ax_lr.set_title('Neighborhood r')

    ax_ul.axis('off')
    ax_ur.axis('off')
    ax_ll.axis('off')
    ax_lr.axis('off')

    def update_function():

        cur_n_iterations = 1
        tribal_masking_computational_core.spatial_segregation_and_externality(model, cur_n_iterations)

        a1 = model.masking_choice
        a2 = model.types_map
        a3 = model.infection_status
        a4 = model.parameter_maps['average_reciprocal_response_from_masking']

        a2 = np.where(a2 > 0, a2, np.nan)
        im_ul.set_data(a1)
        im_ur.set_data(a2)
        # Hack to get non-infected to be white
        a3[a3 == 0] = np.nan
        im_ll.set_data(a3)
        im_lr.set_data(a4)

        percent_masked = np.sum(np.where(a1 == 1, 1, 0)) / model.n_agents
        percent_masked_annotation.set_text('Percent masked: ' + str(hb.round_significant_n(percent_masked, 4)))

        fig.canvas.draw_idle()

    # Define an action for modifying the line when any slider's value changes
    def sliders_on_changed(val):
        print(1234, sliders['segregation_threshold'].val)
        model.params['segregation_threshold'] = sliders['segregation_threshold'].val
        model.params['infection_probability'] = sliders['infection_probability'].val
        model.params['neighborhood_radius'] = sliders['neighborhood_radius'].val
        model.params['b'] = sliders['b'].val
        model.params['d'] = sliders['d'].val
        model.params['s'] = sliders['s'].val
        model.params['r'] = sliders['r'].val

        if model.show_uncertainty_sliders:
            model.params['bd'] = bd_slider.val
            model.params['dd'] = dd_slider.val
            model.params['sd'] = sd_slider.val
            model.params['rd'] = rd_slider.val

        # Because this changes the distribution of the whole population, it requires total redraw of the parameters' random values
        model.initialize_random_parameters()

    def on_mouseclick(event):
        global plot_focused

        if event.xdata is not None and plot_focused:

            if model.debug_level > 10:
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))

            r = round(event.ydata)
            c = round(event.xdata)

            val = model.types_map[r, c]
            print('Infected agent at ' + str(r) + ', ' + str(c) + ' who had type: ' + str(val))

            model.infection_status[r, c] = 1
            # model.parameter_maps['r'][r, c] = 3.0
            # model.parameter_maps['d'][r, c] = 2.0
            # model.types_map[r, c] = 1.0

    fig.canvas.mpl_connect('button_press_event', on_mouseclick)

    # import matplotlib.pyplot as plt

    # Only have key commands work if you're hovering over a plot.
    def on_enter_axes(event):
        global plot_focused

        if len(event.inaxes.title.get_text()) > 5:
            plot_focused = True


    def on_leave_axes(event):
        global plot_focused

        if len(event.inaxes.title.get_text()) > 5:
            plot_focused = False

    fig.canvas.mpl_connect('axes_enter_event', on_enter_axes)
    fig.canvas.mpl_connect('axes_leave_event', on_leave_axes)

    # plt.show()

    def on_keypress(event):
        global plot_focused

        if plot_focused:

            r = round(event.ydata)
            c = round(event.xdata)

            if event.key == 'a':
                model.types_map[r, c] = 1.0
            if event.key == 'q':
                model.types_map[r, c] = 2.0
            if event.key == 's':
                model.parameter_maps['r'][r, c] += .5
            if event.key == 'w':
                model.parameter_maps['r'][r, c] -= .5
            if event.key == 'd':
                model.parameter_maps['d'][r, c]  += .5
            if event.key == 'e':
                model.parameter_maps['d'][r, c] -= .5

    fig.canvas.mpl_connect('key_press_event', on_keypress)


    timer = fig.canvas.new_timer(interval=10)
    timer.add_callback(update_function)
    timer.start()

    # Turn off hover text (was breaking the window size)
    ax_ul.format_coord = lambda x, y: ""
    ax_ur.format_coord = lambda x, y: ""
    ax_ll.format_coord = lambda x, y: ""
    ax_lr.format_coord = lambda x, y: ""

    # def update_textx():
    #
    #     print('this')
    #
    # def submit_text(input_text):
    #     print('input_text', input_text)
    #     # ydata = eval(text)
    #     # l.set_ydata(ydata)
    #     # ax.set_ylim(np.min(ydata), np.max(ydata))
    #     # plt.draw()
    # # Text box to input x value
    # axbox1 = fig.add_axes([0.1, 0.1, 0.5, 0.05])
    # x_textbox = TextBox(axbox1, "New x value")
    # x_textbox.on_submit(update_textx)
    #
    # # Submit button
    # axbox3 = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    # submit_button = Button(axbox3, "Submit!")
    # submit_button.on_clicked(submit_text)



    # Add Sliders
    # Define an axes area and draw a1 slider in it
    slider_properties = {}
    slider_properties['segregation_threshold'] = {'min': 0., 'max': 1., 'init': .3}
    slider_properties['neighborhood_radius'] = {'min': 1., 'max': 6., 'init': 3.}
    slider_properties['infection_probability'] = {'min': 0.0, 'max': .2, 'init': .03}
    slider_properties['b'] = {'min': -2., 'max': 2., 'init': -.1}
    slider_properties['d'] = {'min': -5., 'max': 5., 'init': -.25}
    slider_properties['s'] = {'min': -2, 'max': 2, 'init': 1.}
    slider_properties['r'] = {'min': -5., 'max': 5., 'init': 3.}


    if model.show_uncertainty_sliders:
        slider_properties['bd'] = {'min': -2., 'max': 2.}
        slider_properties['dd'] = {'min': -2., 'max': 2.}
        slider_properties['sd'] = {'min': .00001, 'max': 1.}
        slider_properties['rd'] = {'min': -5., 'max': 5.}

    i = 0
    sliders = {}
    for name, properties in slider_properties.items():
        slider_ax = fig.add_axes([.3, 0.30 - i, 0.4, 0.03], facecolor=axis_color)
        slider = Slider(slider_ax, name, properties['min'], properties['max'], valinit=properties['init'])
        slider.on_changed(sliders_on_changed)
        sliders[name] = slider
        i += .03

    # threshold_slider_ax = fig.add_axes([.3, 0.30, 0.4, 0.03], facecolor=axis_color)
    # threshold_slider = Slider(threshold_slider_ax, 'threshold', 0.0, 1.0, valinit=model.params['segregation_threshold'])
    # threshold_slider.on_changed(sliders_on_changed)
    #
    # neighborhood_radius_slider_ax = fig.add_axes([.3, 0.27, 0.4, 0.03], facecolor=axis_color)
    # neighborhood_radius_slider = Slider(neighborhood_radius_slider_ax, 'neighborhood_radius', 1., 6., valinit=model.params['neighborhood_radius'])
    # neighborhood_radius_slider.on_changed(sliders_on_changed)
    #
    # b_slider_ax = fig.add_axes([.3, 0.24, 0.4, 0.03], facecolor=axis_color)
    # b_slider = Slider(b_slider_ax, 'b', -2., 2., valinit=model.params['b'])
    # b_slider.on_changed(sliders_on_changed)
    #
    # if model.show_uncertainty_sliders:
    #     bd_slider_ax = fig.add_axes([.3, 0.21, 0.4, 0.03], facecolor=axis_color)
    #     bd_slider = Slider(bd_slider_ax, 'bd', -2., 2., valinit=model.params['bd'])
    #     bd_slider.on_changed(sliders_on_changed)
    #
    # d_slider_ax = fig.add_axes([0.3, 0.18, 0.4, 0.03], facecolor=axis_color)
    # d_slider = Slider(d_slider_ax, 'd', -5., 5., valinit=model.params['d'])
    # d_slider.on_changed(sliders_on_changed)
    #
    # if model.show_uncertainty_sliders:
    #     dd_slider_ax = fig.add_axes([0.3, 0.15, 0.4, 0.03], facecolor=axis_color)
    #     dd_slider = Slider(dd_slider_ax, 'dd', 0.00001, 5., valinit=model.params['dd'])
    #     dd_slider.on_changed(sliders_on_changed)
    #
    # s_slider_ax = fig.add_axes([0.3, 0.12, 0.4, 0.03], facecolor=axis_color)
    # s_slider = Slider(s_slider_ax, 's', .0000001, 4., valinit=model.params['s'])
    # s_slider.on_changed(sliders_on_changed)
    #
    # if model.show_uncertainty_sliders:
    #     sd_slider_ax = fig.add_axes([0.3, 0.09, 0.4, 0.03], facecolor=axis_color)
    #     sd_slider = Slider(sd_slider_ax, 'sd', 0.0000001, 1., valinit=model.params['sd'])
    #     sd_slider.on_changed(sliders_on_changed)
    #
    # r_slider_ax = fig.add_axes([0.3, 0.06, 0.4, 0.03], facecolor=axis_color)
    # r_slider = Slider(r_slider_ax, 'r', -5, 5, valinit=model.params['r'])
    # r_slider.on_changed(sliders_on_changed)
    #
    # if model.show_uncertainty_sliders:
    #     rd_slider_ax = fig.add_axes([0.3, .03, 0.4, 0.03], facecolor=axis_color)
    #     rd_slider = Slider(rd_slider_ax, 'rd', 0.0000001, 1., valinit=model.params['rd'])
    #     rd_slider.on_changed(sliders_on_changed)

    plt.show()

