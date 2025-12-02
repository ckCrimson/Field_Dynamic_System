from Research_paper_systems_implementation.dice_system_implementation.dynamics.multi_step.multi_step_field_generator import \
    DiceMultiStepFieldGenerator
from Research_paper_systems_implementation.dice_system_implementation.dynamics.operator.dice_operator import \
    DiceOperator
from Research_paper_systems_implementation.dice_system_implementation.dynamics.single_step.single_step_field_generator import \
    DiceSingleStepFieldGenerator
from Research_paper_systems_implementation.dice_system_implementation.field.dice_field_value import DiceRealFieldValue
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_field import DiceRealField
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_single_field_value import \
    DiceRealSingleFieldValue
from Research_paper_systems_implementation.dice_system_implementation.field.dice_single_field_value_addition import \
    DiceFieldValueAddition
from Research_paper_systems_implementation.dice_system_implementation.field.dice_single_field_value_norm import \
    DiceFieldValueNorm
from Research_paper_systems_implementation.dice_system_implementation.state.dice_multi_step_reaching import \
    DiceMultiStepReaching
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reachable import DiceReachable
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reaching import DiceReaching
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace

# ---------------Constants --------------------#
ALPHA = 2
N_0=10
D=6
# --------- State ---------------#

mu_0 = DiceState(1.2)
print("Initial State: ",mu_0)

#-------------- State Space --------------#

dice_space = DiceStateSpace(mu_0,D,1,6)
print("Dice Space",dice_space)

# -----------Reachable --------------------#

dice_reachable = DiceReachable(ALPHA,N_0,D)
reachable_space = dice_reachable.get_reachable(mu_0)
print("reachable Space: ",dice_reachable.get_reachable(mu_0))

#--------------- Union Intersection -------------#

union_space = reachable_space.union_state_space(dice_space)
print("Union Space: ",union_space)
intersection_space = reachable_space.intersection_state_space(dice_space)
print("Intersection Space: ",intersection_space)

# -------------- Dice Reaching --------------------#

dice_reaching  =  DiceReaching(ALPHA,N_0,D)
reaching_space = dice_reaching.get_reaching(mu_0)
print("Reaching Space: ",dice_reaching.get_reaching(mu_0))

# ------------- Dice MultiStep reaching -----------------#

dice_multi_step_reaching = DiceMultiStepReaching(ALPHA,N_0,D)
multi_step_reaching_space = dice_multi_step_reaching.get_multi_step_reaching(mu_0,1)
print("Multi Step Reaching space",multi_step_reaching_space)

# -------- Dice Single Field Value ------------#

dice_single_field_value = DiceRealSingleFieldValue(1)
print("Dice Single Field Value: ",dice_single_field_value)

# ---------- Dice Field Value ---------------#

DiceRealFieldValue.configure(DiceFieldValueNorm(),DiceFieldValueAddition())

dice_real_field_value = DiceRealFieldValue(DiceRealSingleFieldValue(2))
dice_real_field_value2 = DiceRealFieldValue(DiceRealSingleFieldValue(1))
print("Dice real Field Value: ",dice_real_field_value.data)
print("Dice real Field Value Zero", dice_real_field_value.get_zero_field().data)
print("Dice real Field Value One", dice_real_field_value.get_unitary_field().data)
print("2+1: ",dice_real_field_value2.addition(dice_real_field_value).data)

#-------------Dice Field -------------------#

dice_field= DiceRealField(DiceStateSpace(mu_0,D))

#--------------- Dice Single Step --------------#

dice_single_step_generator = DiceSingleStepFieldGenerator(alpha=ALPHA,n_0=N_0,dice_number=D)
sigle_step_field = dice_single_step_generator.build_single_step_field(mu_0,dice_field)
#sigle_step_field.plot_field("Single Step Field")

#---------------Multi Step Field----------------#

dice_multi_step_field_generator = DiceMultiStepFieldGenerator(alpha=ALPHA,n_0=N_0,dice_number=D)
multi_step_dice_field = dice_multi_step_field_generator.generate_multi_step_field(dice_space,mu_0,2)
#multi_step_dice_field.plot_field("Multi Step Field")

#--------- Dice Random Operator ------------------#

dice_operator = DiceOperator()
new_state = dice_operator.get_next_state(multi_step_dice_field)
print(f"prev_state:{mu_0} ---> new_state:{new_state}")

