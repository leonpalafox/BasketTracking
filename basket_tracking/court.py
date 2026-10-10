"""FIBA half-court geometry in meters, for mapping positions onto a top-down court.

Coordinates: x runs along the baseline from the left sideline (0 to 15 m), y runs from
the baseline toward half court (0 to 14 m). Seen from the stands behind half court,
this matches the camera view: baseline at the top, left sideline on the left.
"""

import numpy as np

WIDTH, HALF_LENGTH = 15.0, 14.0
CX = WIDTH / 2
BASKET_Y = 1.575          # baseline to center of the ring
LANE_W, FT_Y = 4.9, 5.8   # lane width, baseline to free-throw line
FT_R = 1.8                # free-throw circle / center circle radius
THREE_R, THREE_SIDE = 6.75, 0.9
RESTRICTED_R = 1.25


def _arc(cx, cy, r, a0, a1, n=64):
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def court_polylines():
    """List of Nx2 polylines (meters) for the half-court markings."""
    lane_x0, lane_x1 = CX - LANE_W / 2, CX + LANE_W / 2
    # The 3-point line is straight along the sidelines until it meets the arc.
    corner_y = BASKET_Y + np.sqrt(THREE_R**2 - (CX - THREE_SIDE) ** 2)
    a = np.arcsin((corner_y - BASKET_Y) / THREE_R)
    three = np.r_[
        [[THREE_SIDE, 0]],
        _arc(CX, BASKET_Y, THREE_R, np.pi - a, a, 96),
        [[WIDTH - THREE_SIDE, 0]],
    ]
    return [
        np.array([[0, 0], [WIDTH, 0], [WIDTH, HALF_LENGTH], [0, HALF_LENGTH], [0, 0]]),
        np.array([[lane_x0, 0], [lane_x0, FT_Y], [lane_x1, FT_Y], [lane_x1, 0]]),
        _arc(CX, FT_Y, FT_R, 0, 2 * np.pi),
        three,
        _arc(CX, HALF_LENGTH, FT_R, np.pi, 2 * np.pi),
        _arc(CX, BASKET_Y, RESTRICTED_R, 0, np.pi),
        _arc(CX, BASKET_Y, 0.225, 0, 2 * np.pi, 24),            # ring
        np.array([[CX - 0.9, 1.2], [CX + 0.9, 1.2]]),           # backboard
    ]


def draw_court(ax, color="#5b5b57", lw=1.2):
    for line in court_polylines():
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=3)
    ax.set_xlim(-0.5, WIDTH + 0.5)
    ax.set_ylim(HALF_LENGTH + 0.5, -0.5)  # baseline at the top, like the camera view
    ax.set_aspect("equal")
    ax.set_axis_off()
