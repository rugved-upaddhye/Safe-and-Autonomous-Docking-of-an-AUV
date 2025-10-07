extends CharacterBody3D

@export var dock_target: Node3D

const SPEED = 5.0
var is_docking = true # Set to true to run the autonomous logic

func _process(_delta):
	if not is_docking or dock_target == null:
		# Manual control logic (from yesterday)
		# We'll leave it here in case you want to switch back
		# For now, it won't run because is_docking is true
		return

	# --- Autonomous Docking Logic ---
	
	# 1. Calculate Direction
	# Get the position of the dock and this AUV in world space
	var target_position = dock_target.global_position
	var auv_position = self.global_position
	
	# The direction is a simple vector subtraction
	var direction_to_dock = target_position - auv_position
	
	# 2. Stop When Close
	# Check the distance. If it's less than 0.1 meters, we've arrived.
	if direction_to_dock.length() < 0.1:
		velocity = Vector3.ZERO # Stop moving
		print("Docking successful!")
	else:
		# 3. Move the AUV
		# We normalize the direction to get a unit vector (length of 1), then multiply by speed.
		velocity = direction_to_dock * 0.5
		#velocity = direction_to_dock.normalized() * SPEED
	
		# 4. Face the Target
		# This one-line command makes the AUV's -Z axis (its "front") point at the target.
		look_at(target_position)

	# Move the AUV using the calculated velocity
	move_and_slide()
