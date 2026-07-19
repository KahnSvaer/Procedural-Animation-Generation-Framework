import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import torch

from animgen.core.spline import Spline


points = [] # tensor([x, y, 0.0])
curve_points = []

spline_generated = False # UI Flag to allow for adding points before spline generation and locking editing after.



def redraw():
    """
    Redraw the complete current state of the experiment.
    """

    ax.clear()

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.grid(True)

    if not spline_generated:
        ax.set_title(
            "Spline Test — Editing Control Points\n"
            "Left Click: Add Point | "
            "Right Click: Undo | "
            "Enter: Generate Spline"
        )

    else:
        ax.set_title(
            "Spline Test — Spline Generated\n"
            "Point Editing Locked | "
            "Enter: Reset"
        )

    if points:
        control_x = [
            point[0].item()
            for point in points
        ]

        control_y = [
            point[1].item()
            for point in points
        ]

        ax.scatter(
            control_x,
            control_y,
            label="Control Points",
        )

        ax.plot(
            control_x,
            control_y,
            linestyle="--",
            label="Control Polygon",
        )

    if spline_generated and curve_points:
        curve_x = [
            point[0].item()
            for point in curve_points
        ]

        curve_y = [
            point[1].item()
            for point in curve_points
        ]

        ax.plot(
            curve_x,
            curve_y,
            linewidth=2,
            label="Catmull-Rom Spline",
        )

    if points:
        ax.legend()

    fig.canvas.draw_idle()


def onclick(event):
    """
    Handle mouse input.

    Left Click:
        Add a new 3D control point with z = 0.

    Right Click:
        Remove the most recently added control point.

    Mouse input is disabled after spline generation.
    """

    if event.inaxes is None:
        return

    if spline_generated:
        print(
            "Spline has already been generated. "
            "Press Enter to reset."
        )
        return

    if event.button == 1:
        point = torch.tensor(
            [
                float(event.xdata),
                float(event.ydata),
                0.0,
            ],
            dtype=torch.float32,
        )

        points.append(point)

        print(f"Added: {point}")

        redraw()

    elif event.button == 3 and points:
        removed = points.pop()

        print(f"Removed: {removed}")

        redraw()


def onkey(event):
    """
    Handle keyboard input.

    Enter:
        Editing mode   -> Generate spline.
        Generated mode -> Reset everything.
    """

    global spline_generated
    global curve_points

    if event.key != "enter":
        return

    if spline_generated:
        print("\nResetting spline test.")

        points.clear()
        curve_points.clear()

        spline_generated = False

        redraw()

        return

    if len(points) < 3:
        print(
            "\nAt least 3 control points are required "
            "to generate the spline."
        )
        return

    print(
        "\nGenerating spline with the following "
        "control points:"
    )

    for point in points:
        print(point)

    # Create and evaluate spline from 3D tensor control points.
    spline = Spline(points)
    curve_points = spline.evaluate_curve(
        num_points_per_segment=100
    )

    spline_generated = True

    redraw()


fig, ax = plt.subplots()

redraw()


fig.canvas.mpl_connect(
    "button_press_event",
    onclick,
)

fig.canvas.mpl_connect(
    "key_press_event",
    onkey,
)


plt.show()