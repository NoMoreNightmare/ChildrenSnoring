import os.path
from configparser import ConfigParser
from time import sleep

import win32api
import win32con
import win32gui
import win32process
import pyautogui

from win32con import WM_INPUTLANGCHANGEREQUEST


def change_language(language="EN"):
    """
    切换语言
    :param language: EN––English; ZH––Chinese
    :return: bool
    """
    LANGUAGE = {
        "ZH": 0x0804,
        "EN": 0x0409
    }
    hwnd = win32gui.GetForegroundWindow()
    language = LANGUAGE[language]
    win32api.SendMessage(hwnd, WM_INPUTLANGCHANGEREQUEST, 0, language)


def mouse_click(x, y):
    win32api.SetCursorPos((x, y))

    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)

    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


def open_software():
    sleep(1)
    config = ConfigParser()
    config.read('property.cfg')
    software = config.get("software", "software")
    exe_name = r"{}".format(software)
    si = win32process.STARTUPINFO()
    si.dwFlags = win32con.STARTF_USESHOWWINDOW
    si.wShowWindow = win32con.SW_MAXIMIZE
    win32process.CreateProcess(None, exe_name, None, None, False, 0, None, None, si)


def open_mp3file(mp3):
    # click file
    sleep(3)
    mouse_click(22, 30)
    # click open-location
    sleep(1)
    mouse_click(61, 122)
    # change chinese keyboard into english
    sleep(1)
    change_language("EN")
    # input the path of mp3 file
    sleep(0.5)
    for i in mp3:
        pyautogui.press(i)

    return os.path.basename(mp3)


def save_file(imageName):
    # click file
    sleep(1)
    mouse_click(22, 30)
    # click export as image
    sleep(1)
    mouse_click(149, 447)
    # get image name
    sleep(1)
    path = "images"
    path = os.path.realpath(path)

    mouse_click(154, 91)
    sleep(1)
    for i in path:
        pyautogui.press(i)

    sleep(1)
    pyautogui.press('enter')
    sleep(1)
    mouse_click(328, 497)
    sleep(1)
    imageName = imageName + ".png"
    for i in imageName:
        pyautogui.press(i)

    # click save
    sleep(1)
    mouse_click(755, 591)
    if os.path.exists(path + "/" + imageName):
        sleep(1)
        mouse_click(1044, 593)
        sleep(0.5)
        mouse_click(1059, 605)
    else:
        # click OK
        sleep(1)
        mouse_click(1063, 616)
        sleep(1)


def close_software():
    sleep(1)
    mouse_click(1888, 9)
    sleep(2)
    mouse_click(983, 595)


def automatic_manipulation(mp3):
    sleep(2)
    open_software()
    sleep(1)
    imageName = open_mp3file(mp3)
    # click layer
    sleep(0.5)
    mouse_click(959, 577)
    # click add spectrum
    sleep(0.5)
    mouse_click(217, 41)
    # click mixed
    sleep(0.5)
    mouse_click(355, 290)
    sleep(0.5)
    mouse_click(657, 296)
    # change color
    sleep(0.5)
    mouse_click(1764, 155)
    sleep(0.5)
    mouse_click(1729, 197)
    # change scale
    sleep(0.5)
    mouse_click(1713, 197)
    sleep(0.5)
    mouse_click(1724, 226)
    sleep(0.5)
    mouse_click(1788, 192)
    sleep(0.5)
    mouse_click(1814, 237)

    sleep(0.5)
    win32api.SetCursorPos((1866, 191))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 1866, 191, 0, 0)
    win32api.SetCursorPos((1888, 193))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 1888, 193, 0, 0)

    save_file(imageName)
    close_software()


if __name__ == "__main__":
    # read property file
    config = ConfigParser()
    config.read('property.cfg')
    # get the path of mp3 file
    # mp3 = config.get("mp3files", "mp3file1")
    mp3files = config.options("mp3files")
    for i in mp3files:
        record_files = config.get("mp3files", i)
        if os.path.isdir(record_files):
            files = os.listdir(record_files)
            for file in files:
                record_path = record_files + "/" + file
                # print(record_path)
                automatic_manipulation(record_path)
        else:
            automatic_manipulation(record_files)

    # while True:
    #     sleep(3)
    #     print(win32api.GetCursorPos())
