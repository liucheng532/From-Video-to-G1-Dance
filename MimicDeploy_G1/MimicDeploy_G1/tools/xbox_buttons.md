pressButton(IntEnum):
A = 0                    ✓
B = 1                    ✓
X = 2                    ✓
Y = 3                    ✓
LB(L1) = 4               ✓
RB(R1) = 5               ✓
SELECT(Left) = 6         ✓
START(Right) = 7         ✓
XHOME(Up) = 8            -> L3 = 8     # Left Stick Press
SHARE(Down) = 11         -> UP = 11    # D-pad Up (if mapped as separate button)
LEFT_AXIS_PRESS = 9      -> R3 = 9     # Right Stick Press
RIGHT_AXIS_PRESS = 10    -> HOME = 10  # PS: PS FSMCommand, Xbox: Xbox FSMCommand

axisButton:
LT(L2) = 2 [press -> 1.0] [init -> -1.0]
RT(R2) = 5 [press -> 1.0] [init -> -1.0]

Unitree R3 遥控器
JoystickButton(IntEnum):
# Standard PlayStation/Xbox Layout
A = 0      # PS: Cross(×), Xbox: A
B = 1      # PS: Circle(○), Xbox: B
X = 2      # PS: Square(□), Xbox: X
Y = 3      # PS: Triangle(△), Xbox: Y
L1 = 4     # Left Bumper (L1 on PS)
R1 = 5     # Right Bumper (R1 on PS)
SELECT = 6   # Select/Share button
START = 7  # Start/Options button
L3 = 8     # Left Stick Press
R3 = 9     # Right Stick Press
HOME = 10  # PS: PS FSMCommand, Xbox: Xbox FSMCommand
UP = 11    # D-pad Up (if mapped as separate button)
DOWN = 12  # D-pad Down
LEFT = 13  # D-pad Left
RIGHT = 14 # D-pad Right


