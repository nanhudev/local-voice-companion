class_name VoiceCompanionUI
extends CanvasLayer

const GATEWAY := "http://127.0.0.1:17831"

@onready var status_dot: ColorRect = $Panel/Margin/VBox/Header/StatusDot
@onready var status_label: Label = $Panel/Margin/VBox/Header/Status
@onready var user_label: Label = $Panel/Margin/VBox/UserText
@onready var assistant_label: Label = $Panel/Margin/VBox/AssistantText
@onready var health_request: HTTPRequest = $HealthRequest
@onready var events_request: HTTPRequest = $EventsRequest
@onready var control_request: HTTPRequest = $ControlRequest
@onready var poll_timer: Timer = $PollTimer

var _last_event_id := 0
var _polling := false
var _enabled := true


func _ready() -> void:
	if DisplayServer.get_name() == "headless":
		visible = false
		return
	health_request.request_completed.connect(_on_health_completed)
	events_request.request_completed.connect(_on_events_completed)
	control_request.request_completed.connect(_on_control_completed)
	poll_timer.timeout.connect(_poll_events)
	_set_status("正在连接本地语音…", Color(0.95, 0.72, 0.25))
	health_request.request(GATEWAY + "/health")


func _unhandled_input(event: InputEvent) -> void:
	if not event.is_action_pressed("voice_toggle"):
		return
	_enabled = not _enabled
	var headers := PackedStringArray(["Content-Type: application/json"])
	control_request.request(GATEWAY + "/control", headers, HTTPClient.METHOD_POST, JSON.stringify({"enabled": _enabled}))
	get_viewport().set_input_as_handled()


func _poll_events() -> void:
	if _polling:
		return
	_polling = true
	var error := events_request.request(GATEWAY + "/events?after=" + str(_last_event_id))
	if error != OK:
		_polling = false


func _on_health_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code != 200:
		_set_status("语音网关离线", Color(0.95, 0.30, 0.25))
		return
	var payload = JSON.parse_string(body.get_string_from_utf8())
	if payload is not Dictionary:
		return
	_enabled = payload.get("enabled", true)
	_apply_state(payload.get("state", "listening"), "")
	poll_timer.start()


func _on_events_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	_polling = false
	if response_code != 200:
		_set_status("语音连接中断", Color(0.95, 0.30, 0.25))
		return
	var payload = JSON.parse_string(body.get_string_from_utf8())
	if payload is not Dictionary:
		return
	for event in payload.get("events", []):
		_last_event_id = maxi(_last_event_id, int(event.get("id", 0)))
		var event_type: String = event.get("type", "")
		var text: String = event.get("text", "")
		var state: String = event.get("state", "")
		if event_type == "user":
			user_label.text = "你：" + text
		elif event_type in ["assistant", "assistant_partial"]:
			assistant_label.text = "助手：" + text
		elif event_type == "error":
			_set_status("语音错误：" + event.get("detail", "未知错误"), Color(0.95, 0.30, 0.25))
		if not state.is_empty():
			_apply_state(state, text)


func _on_control_completed(_result: int, response_code: int, _headers: PackedStringArray, _body: PackedByteArray) -> void:
	if response_code != 200:
		_set_status("切换失败：语音网关离线", Color(0.95, 0.30, 0.25))


func _apply_state(state: String, text: String) -> void:
	match state:
		"listening": _set_status("正在聆听（V 暂停）", Color(0.25, 0.92, 0.52))
		"hearing": _set_status("听到了，请继续说…", Color(0.25, 0.72, 1.0))
		"transcribing": _set_status("正在识别语音…", Color(0.35, 0.70, 1.0))
		"thinking": _set_status("正在思考…", Color(0.72, 0.48, 1.0))
		"speaking": _set_status("正在回答…", Color(1.0, 0.58, 0.28))
		"paused": _set_status("已暂停（V 继续）", Color(0.58, 0.60, 0.66))
		"error": _set_status(text if not text.is_empty() else "语音服务异常", Color(0.95, 0.30, 0.25))


func _set_status(text: String, color: Color) -> void:
	status_label.text = text
	status_dot.color = color
