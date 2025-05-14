import cython
import numpy as np
cimport numpy as np

DTYPEUINT8 = np.uint8
DTYPEBYTE = np.byte
DTYPEINT = np.int64
DTYPEINT64 = np.int64
DTYPELONG = np.int32
DTYPEFLOAT32 = np.float32
DTYPEFLOAT64 = np.float64
ctypedef np.uint8_t DTYPEUINT8_t
ctypedef np.int32_t DTYPEINT_t
ctypedef np.int64_t DTYPEINT64_t
ctypedef np.float32_t DTYPEFLOAT32_t
ctypedef np.float64_t DTYPEFLOAT64_t

from libc.stdlib cimport rand, RAND_MAX
cdef float RAND_MAX_float = <float>(RAND_MAX)

@cython.cdivision(True)
@cython.embedsignature(True)
@cython.boundscheck(True)
@cython.wraparound(True)
cpdef float[::, ::1] spatial_externality_game(float[::, ::1] direct_benefit, float[::, ::1] social_benefit, float[::, ::1] reciprocal_response):
    cdef long n_rows = direct_benefit.shape[0]
    cdef long n_cols = direct_benefit.shape[1]
    cdef long r, c, rd, cd
    cdef float utility_from_masking, utility_from_not_masking

    cdef float[::, ::1] output_array = np.empty([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef float[::, ::1] masking_choice = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef float[::, ::1] average_reciprocal_response_from_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef float[::, ::1] average_reciprocal_response_from_not_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)


    for r in range(n_rows):
        for c in range(n_cols):

            output_array[r, c] = direct_benefit[r, c] + social_benefit[r, c] + reciprocal_response[r, c]

            for rd in range(-1, 1):
                for cd in range(-1, 1):
                    # Possible optimization: remove testing for own cell by smartly building tuples of which to test. Might also be done when considering non d8 neighbors.

                    if not (rd == 0 and cd == 0): # Make sure to not account for one's reciprocal response to themself.
                        average_reciprocal_response_from_masking[r, c] += reciprocal_response[r + rd, c + cd]
                        average_reciprocal_response_from_not_masking[r, c] += -1 * reciprocal_response[r + rd, c + cd]

            average_reciprocal_response_from_masking[r, c] = average_reciprocal_response_from_masking[r, c] / 8.
            utility_from_masking = direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_masking[r, c] # Note implicit masking_choice as * 1
            utility_from_not_masking = -1 * direct_benefit[r, c] + social_benefit[r, c] * average_reciprocal_response_from_not_masking[r, c]

            if utility_from_masking > utility_from_not_masking:
                masking_choice[r, c] = 1
            else:
                masking_choice[r, c] = -1

    return masking_choice

@cython.cdivision(True)
@cython.embedsignature(True)
@cython.boundscheck(True)
@cython.wraparound(True)
def update_model_arrays(model, long n_iterations):
    

    # Extract model attributes to c-objects
    cdef np.ndarray[np.int32_t, ndim=1] agent_ids = model.agent_ids
    cdef np.ndarray[np.int32_t, ndim=2] agent_locations = model.agent_locations
    cdef np.ndarray[np.int32_t, ndim=2] unoccupied_locations = model.unoccupied_locations
    cdef np.ndarray[np.int32_t, ndim=1] agent_types = model.agent_types
    cdef np.ndarray[np.int32_t, ndim=2] agent_ids_map = model.agent_ids_map
    cdef np.ndarray[np.int32_t, ndim=2] types_map = model.types_map

    cdef np.ndarray[np.float32_t, ndim=2] d_map = model.parameter_maps['d']
    cdef np.ndarray[np.float32_t, ndim=2] s_map = model.parameter_maps['s']
    cdef np.ndarray[np.float32_t, ndim=2] r_map = model.parameter_maps['r']
    cdef np.ndarray[np.float32_t, ndim=2] b_map = model.parameter_maps['b']
    cdef np.ndarray[np.float32_t, ndim=2] masking_choice = model.masking_choice
    cdef np.ndarray[np.float32_t, ndim=2] infection_status = model.infection_status
    cdef np.ndarray[np.float32_t, ndim=2] immunity_status = model.immunity_status
    cdef np.ndarray[np.float32_t, ndim=2] immunity_efficacy = model.immunity_efficacy

    cdef float  infection_duration = model.params['infection_duration']
    cdef float immunity_decay = model.params['immunity_decay']

    cdef float segregation_threshold = model.params['segregation_threshold']
    cdef float infection_probability = model.params['infection_probability']
    cdef float masking_efficacy = model.params['masking_efficacy']

    cdef long neighborhood_radius = model.params['neighborhood_radius']
    cdef str game_type = model.game_type
    cdef long reporting_threshold = 0

    # Define internal c vars
    cdef long n_agents = len(agent_ids)
    cdef long n_rows = agent_ids_map.shape[0]
    cdef long n_cols = agent_ids_map.shape[1]
    cdef long n_cells = n_rows * n_cols
    cdef long iteration_counter, random_unoccupied_index, decision_to_move
    cdef long r, c, rd, cd, neighborhood_r, neighborhood_c, new_r, new_c

    cdef float sum_mismatch, mean_mismatch, sum_match, mean_match, sum_metric, mean_metric, n_in_neighborhood
    cdef float utility_from_masking, utility_from_not_masking, current_infection_probability

    # This is all wrong.  should be initialized.
    cdef np.ndarray[np.int32_t, ndim=2] occupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    cdef np.ndarray[np.int32_t, ndim=2] unoccupied_map = np.zeros([n_rows, n_cols], dtype=DTYPELONG)
    cdef np.ndarray[np.float32_t, ndim=2] mean_similarity_metric_map = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef np.ndarray[np.float32_t, ndim=2] average_reciprocal_response_from_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    cdef np.ndarray[np.float32_t, ndim=2] sum_reciprocal_response_from_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    # cdef float[::, ::1] utility_from_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    # cdef float[::, ::1] utility_from_not_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    # cdef float[::, ::1] utility_from_not_masking = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)
    # cdef float[::, ::1] masking_choice = np.zeros([n_rows, n_cols], dtype=DTYPEFLOAT32)



    cdef long n_unoccupied_cells = n_cells - n_agents

    cdef long n_changers = 0

    n_in_neighborhood = <float> (neighborhood_radius * 2 + 1) ** 2 - 1

    for iteration_counter in range(n_iterations):
        n_changers = 0
        for agent_id in range(n_agents):

            sum_metric = 0
            r = agent_locations[agent_id, 0]
            c = agent_locations[agent_id, 1]
            for rd in range(-neighborhood_radius, neighborhood_radius + 1):
                for cd in range(-neighborhood_radius, neighborhood_radius + 1):
                    neighborhood_r = r + rd
                    neighborhood_c = c + cd
                    if neighborhood_r >= n_rows:
                        neighborhood_r -= n_rows
                    if neighborhood_c >= n_cols:
                        neighborhood_c -= n_cols
                    if game_type == 'move_if_not_enough_similar':
                        if types_map[r, c] == types_map[neighborhood_r, neighborhood_c]:
                            if not (neighborhood_r == r and neighborhood_c == c): # Make sure to not account for one's reciprocal response to themself.
                                sum_metric += 1
                    elif game_type == 'move_if_too_many_dissimilar':
                        if types_map[r, c] != types_map[neighborhood_r, neighborhood_c]:
                            if not (neighborhood_r == r and neighborhood_c == c): # Make sure to not account for one's reciprocal response to themself.
                                sum_metric += 1


            mean_metric = sum_metric / n_in_neighborhood
            # if agent_id == 3:
            #     print('cd', rd, cd, r, c, types_map[r, c], types_map[neighborhood_r, neighborhood_c], sum_metric, mean_metric)

            if game_type == 'move_if_not_enough_similar':
                if mean_metric < segregation_threshold:
                    decision_to_move = 1
                else:
                    decision_to_move = 0
            elif game_type == 'move_if_too_many_dissimilar':
                if mean_metric >= segregation_threshold:
                    decision_to_move = 1
                else:
                    decision_to_move = 0

            mean_similarity_metric_map[r, c] = mean_metric

            if decision_to_move == 1:
                n_changers += 1
                random_unoccupied_index = np.random.randint(0, n_unoccupied_cells)

                new_r = unoccupied_locations[random_unoccupied_index, 0]
                new_c = unoccupied_locations[random_unoccupied_index, 1]

                agent_locations[agent_id, 0] = new_r
                agent_locations[agent_id, 1] = new_c

                agent_ids_map[new_r, new_c] = agent_id
                agent_ids_map[r, c] = 0

                types_map[new_r, new_c] = types_map[r, c]
                types_map[r, c] = 0

                unoccupied_locations[random_unoccupied_index, 0] = r
                unoccupied_locations[random_unoccupied_index, 1] = c

                d_map[new_r, new_c] = d_map[r, c]
                d_map[r, c] = 0

                s_map[new_r, new_c] = s_map[r, c]
                s_map[r, c] = 0

                r_map[new_r, new_c] = r_map[r, c]
                r_map[r, c] = 0

                b_map[new_r, new_c] = b_map[r, c]
                b_map[r, c] = 0

                masking_choice[r, c] = 0 # Not reset because is recalculated hwne moved.

                infection_status[new_r, new_c] = infection_status[r, c]
                infection_status[r, c] = 0


                sum_metric = 0
                r = agent_locations[agent_id, 0]
                c = agent_locations[agent_id, 1]


            for rd in range(-neighborhood_radius, neighborhood_radius + 1):
                for cd in range(-neighborhood_radius, neighborhood_radius + 1):
                    neighborhood_r = r + rd
                    neighborhood_c = c + cd
                    if neighborhood_r >= n_rows:
                        neighborhood_r -= n_rows
                    if neighborhood_c >= n_cols:
                        neighborhood_c -= n_cols

                    if types_map[r, c] == types_map[neighborhood_r, neighborhood_c]:
                        if not (neighborhood_r == r and neighborhood_c == c):  # Make sure to not account for one's reciprocal response to themself.
                            sum_metric += 1

                    sum_reciprocal_response_from_masking[r, c] += r_map[neighborhood_r, neighborhood_c]

                    if infection_status[neighborhood_r, neighborhood_c] > 0:
                        # print(RAND_MAX, rand(), <float>(rand()) / RAND_MAX)

                        current_infection_probability = infection_probability * (1. - (masking_efficacy * (masking_choice[r, c] / 2. + 1.))) * (1. - immunity_efficacy[r, c])
                        if <float>(rand()) / RAND_MAX_float < current_infection_probability:
                            # print('new infection', r, c)
                            infection_status[r, c] = 1
                            immunity_status[r, c] = 0

            # Advance the stage of the infection
            if infection_status[r, c] > 0:
                infection_status[r, c] += 1
                if infection_status[r, c] > infection_duration:
                    infection_status[r, c] = 0
                    immunity_status[r, c] = 1
            if immunity_status[r, c] > 0:
                immunity_status[r, c] += 1

                immunity_efficacy[r, c] = immunity_decay ** immunity_status[r, c]

            average_reciprocal_response_from_masking[r, c] = sum_reciprocal_response_from_masking[r, c] / n_in_neighborhood



            utility_from_masking = d_map[r, c] + s_map[r, c] * average_reciprocal_response_from_masking[r, c] # Note implicit masking_choice as * 1
            utility_from_not_masking = -1 * d_map[r, c] + -1 * s_map[r, c] * average_reciprocal_response_from_masking[r, c]

            if utility_from_masking > utility_from_not_masking:
                masking_choice[r, c] = 1.
            else:
                masking_choice[r, c] = -1.

        if n_changers <= 0:
            break

    # Place c objects back in python class.
    model.agent_ids = agent_ids
    model.agent_locations = agent_locations
    model.unoccupied_locations = unoccupied_locations
    model.agent_ids_map = agent_ids_map
    model.types_map = types_map
    model.parameter_maps['d'] = d_map
    model.parameter_maps['s'] = s_map
    model.parameter_maps['r'] = r_map
    model.parameter_maps['b'] = b_map
    # model.parameter_maps['immunity_status'] = immunity_status
    model.masking_choice = masking_choice
    model.infection_status = infection_status
    model.immunity_status = immunity_status
    model.mean_similarity_metric_map = mean_similarity_metric_map
    model.parameter_maps['average_reciprocal_response_from_masking'] = average_reciprocal_response_from_masking

