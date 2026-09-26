extends AudioStreamPlayer3D

func play_stream(stream_to_play, volume_to_play):
	stream = stream_to_play
	volume_db = volume_to_play
	play()
	await get_tree().create_timer(stream.get_length()).timeout
	queue_free()