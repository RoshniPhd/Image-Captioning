import PySimpleGUI as sg


def popup(full, pre):
    layout = [[sg.Button(f'{" " * 7}Full-Analysis+Plots{" " * 7}'), sg.Button(f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}')]]

    event, values = sg.Window('', layout).read(close=True)
    if event == f'{" " * 7}Full-Analysis+Plots{" " * 7}':
        full()
    elif event == f'{" " * 3}Plot Pre-Evaluated Data{" " * 3}':
        pre()
