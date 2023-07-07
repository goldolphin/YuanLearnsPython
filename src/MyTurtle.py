from ColabTurtle.Turtle import *
import ipywidgets as widgets
import math

def circle(radius, extent=360, steps=None):
    h = heading()
    ratio = extent/360.0
    whole_steps = 0
    if steps == None:
        whole_steps = radius if radius <= 50 else 50
        steps = int(whole_steps * ratio)
    else:
        whole_steps = steps / ratio
    step_len = radius * math.sin(2*math.pi / whole_steps)
    angle = 360/whole_steps
    left(90 - angle/2)
    for i in range(steps):
        forward(step_len)
        right(angle)
    setheading(h)

def swim(distance, period = distance, amplitude = None, steps = None):
    h = heading()
    amplitude = amplitude or period/2
    if steps == None:
        steps = int((period if period <= 30 else 30) * distance / period)
    step_len = distance / steps
    p = position()
    cos_h = math.cos(h*math.pi/180)
    sin_h = math.sin(h*math.pi/180)
    for i in range(steps):
        xt = i*step_len
        yt = amplitude * math.sin(xt*2*math.pi/period)
        x = xt*cos_h - yt*sin_h
        y = xt*sin_h + yt*cos_h
        setposition(p[0]+x, p[1]+y)
        a = math.atan(math.cos(xt*2*math.pi/period))
        setheading(h + a*180/math.pi)
        
    setheading(h)

def create_controller():
    layout=widgets.Layout(width='60px', height='50px')
    empty_label = widgets.Label('', layout=layout)
    button_up = widgets.Button(description='Up', layout=layout)
    button_down = widgets.Button(description='Down', layout=layout)
    button_left = widgets.Button(description='Left', layout=layout)
    button_right = widgets.Button(description='Right', layout=layout)

    box = widgets.VBox([
        widgets.HBox([empty_label, button_up, empty_label]),
        widgets.HBox([button_left, empty_label, button_right]),
        widgets.HBox([empty_label, button_down, empty_label])
    ])

    display(box)

    button_up.on_click(lambda _: forward(10))
    button_down.on_click(lambda _: backward(10))
    button_left.on_click(lambda _: left(10))
    button_right.on_click(lambda _: right(10))
