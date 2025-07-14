from pipeline import detect_ui_element
import pyautogui


target = "cat with laptop in the background"

result = detect_ui_element(target_item=target, monitor_id=1, mode="point", verbose=True)
print(result['center_point'])

pyautogui.moveTo(result['center_point'])