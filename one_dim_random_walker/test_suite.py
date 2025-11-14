

from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition, \
    RealFieldNormTransform, RealFieldValue, RealSingleFieldValue
from one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching
from one_dim_random_walker.core.states.reachable import OneDimensionReachable
from one_dim_random_walker.core.states.state import IntegerState
from one_dim_random_walker.core.states.state_space import IntegerLine
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem
from one_dim_random_walker.dynamics.multi_step_field import OneDimRandomWalkerMultiStep
from one_dim_random_walker.dynamics.operators import OneDimRandomWalkerExpectationOperator, \
    OneDimRandomWalkerSampleOperator
from one_dim_random_walker.dynamics.single_step import OneDimSingleStep

#------------ Test Parameters ---------------#

number_of_executions = 1
number_of_repeats = 1




#------------- State Test ---------------------#

initial_state = IntegerState(0)

#---------------- State Space Test --------------#

integer_line_1 = IntegerLine(
current_state =  initial_state,
left_limit = -200,
right_limit= 200
)


#-------------Reachable -----------------#

integer_reachable = OneDimensionReachable(6)

#reachable_space = integer_reachable.get_reachable(state=initial_state)

#reachable_space_from_allowed = integer_reachable.get_reachable_from_allowed(state=initial_state,space = integer_line_1)

# ------------Multi Step Reaching ---------------- #

multi_step_reaching = OneDimensionMultiStepReaching(1)
forentier_space = multi_step_reaching.get_multi_step_reaching(initial_state=initial_state,
    reference_space=integer_line_1,
    l=7,  # <-- provide this
    parallel=True  )
#
# for s in forentier_space.get_all_states():
#     print(s)
# 2. Test Code: This is the actual method call you want to time.
# SETUP_CODE = ("from one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine"
#               "\ntest_obj = OneDimensionMultiStepReaching(6)"
#               "\ninitial_state = IntegerState(0)\ninteger_line_1 = IntegerLine(current_state =  initial_state,"
#               "left_limit = -200,"
#               "right_limit= 200)")
# TEST_CODE = "test_obj.get_multi_step_reaching(initial_state, integer_line_1,20)"

# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
#
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

"""

Total test time (best of 5 repeats): 1.7397 seconds
Average time per call (100 runs): 17397.194 microseconds

"""
# -------------  Field  -------------------------  #
field_demo=RealField.from_length(9,initial_state)
#field_demo.plot_field()


#---------------------Field Composition ---------------------#
input_field_1 = RealField.from_length(100,initial_state)
input_field_2 = RealField.from_length(100,initial_state)
addition =   RealFieldAddition()
added_field = addition.apply(input_field_1, input_field_2)

# SETUP_CODE = ("from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition,RealFieldAdditionStateless"
#               "\nfrom one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.reachable import OneDimensionReachable"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine\ninitial_state = IntegerState(0)"
#               "\ninput_field_1 = RealField.from_length(10000,initial_state)\ninput_field_2 = RealField.from_length(20000,initial_state)")
# TEST_CODE = "addition =  RealFieldAddition()\nadded_field = addition.apply(input_field_1, input_field_2)"
# #added_field.plot_field()
# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# #Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
# print("Runnig tests over field composition")
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

#------------------------------------------#

field_norm_transform = RealFieldNormTransform()

current_field=RealField.from_length(100,initial_state)

transformed_field=field_norm_transform.apply(current_field,current_field)
#transformed_field.plot_field()

# SETUP_CODE = ("from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition,RealFieldAdditionStateless,RealFieldNormTransform"
#               "\nfrom one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.reachable import OneDimensionReachable"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine\n"
#               "initial_state = IntegerState(0)"
#               "\nfield_norm_transform = RealFieldNormTransform()\n"
#               "current_field=RealField.from_length(100,initial_state)")
# TEST_CODE = "transformed_field=field_norm_transform.apply(current_field,current_field)"
# #added_field.plot_field()
# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# #Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
# print("Testing for the field norm")
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

# -------------- Single Step -------------------- #

single_step_test = OneDimSingleStep()
system_field_ref = RealField.from_length(1000,initial_state)
single_step_test.build_single_step_field(initial_state,system_field_ref)
# single_step_test.build_single_step_field_new(initial_state,system_field_ref).plot_field()
#
# SETUP_CODE = ("from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField, RealFieldAddition,RealFieldAdditionStateless,RealFieldNormTransform"
#               "\nfrom one_dim_random_walker.core.states.multi_step_reaching import OneDimensionMultiStepReaching"
#               "\nfrom one_dim_random_walker.core.states.reachable import OneDimensionReachable"
#               "\nfrom one_dim_random_walker.core.states.state import IntegerState"
#               "\nfrom one_dim_random_walker.core.states.state_space import IntegerLine\nfrom one_dim_random_walker.dynamics.single_step import OneDimSingleStep\n"
#               "initial_state = IntegerState(0)"
#               "\nsingle_step_test = OneDimSingleStep()\nsystem_field_ref = RealField.from_length(1000,initial_state)\n"
#               "")
# TEST_CODE = "single_step_test.build_single_step_field_new(initial_state,system_field_ref)"
# #added_field.plot_field()
# times = timeit.repeat(
#     stmt=TEST_CODE,
#     setup=SETUP_CODE,
#     number=number_of_executions,
#     repeat=number_of_repeats
# )
# best_total_time = min(times)
#
# #Calculate the average time per method call
# average_time_per_call = best_total_time / number_of_executions
# print("Testing for the single step")
# print(f"Total test time (best of {number_of_repeats} repeats): {best_total_time:.4f} seconds")
# print(f"Average time per call ({number_of_executions} runs): {average_time_per_call * 1000000:.3f} microseconds")

# before optimization

#---------------- Multi_step -----------------#


one_dim_multi_step =  OneDimRandomWalkerMultiStep(distance=1)

space = IntegerLine(current_state=initial_state,states_list=[initial_state])

single_transformed_field_proto = RealField.from_length(1,initial_state)
single_step_transformed_field_proto=RealField.from_length(1,initial_state)
prev_field_input = RealField(space)
prev_field_input.set_zero_field()
prev_field_input.set_field(initial_state,RealFieldValue(RealSingleFieldValue(1)))


multi_field=one_dim_multi_step.generate_multi_step_field(space,initial_state,3,single_transformed_field_proto,single_step_transformed_field_proto,
                                              prev_field_input )
# multi_field.plot_field()
#----------- Operator ------------------#

#
# print("Expectation_operator: " , OneDimRandomWalkerExpectationOperator().get_next_state(multi_field) )
# print("Probabiliostic_operator: " , OneDimRandomWalkerSampleOperator().get_next_state(multi_field) )
#

# -----------Dynamic System ----------------#

sys_field=RealField.from_length(100,initial_state)

sys_field.set_unit_field_at_state(initial_state)

#sys_field.plot_field()



one_dim_walker_fds  = OneDimWalkerFieldDynamicSystem(initial_state,sys_field,OneDimRandomWalkerMultiStep(2),OneDimensionReachable(2),OneDimRandomWalkerSampleOperator())
print(one_dim_walker_fds.transition_list)
one_dim_walker_fds.evolve(2)
print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)
print(one_dim_walker_fds.transition_list)
one_dim_walker_fds.evolve(13)
print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)
print(one_dim_walker_fds.transition_list)
one_dim_walker_fds.evolve(13)
print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)
print(one_dim_walker_fds.transition_list)
print(one_dim_walker_fds.field.get_field(one_dim_walker_fds.initial_state).data.value)