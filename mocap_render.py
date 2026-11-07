"""Render a motion-capture clip as a plain figure with a side-on tracking camera, to use as an Animate motion reference.

Usage: blender -b -P mocap_render.py -- motion.bvh out.mp4 [--seconds 3] [--fps 24] [--speed 1] [--depth]

Long takes are cut to their busiest stretch. Short clips are looped: it finds one full stride, repeats it, and carries
the body forward each time. A moving body gets a side-on tracking camera; one that stays in place gets a still camera
in front, turned 30 degrees.
"""
import argparse
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HEIGHT = 1.7  # metres, so framing doesn't depend on the capture's units
# body part radii as a share of height; bones not listed (fingers, thumbs, end sites) aren't drawn
RADII = {"LHipJoint": .05, "RHipJoint": .05, "LeftUpLeg": .044, "RightUpLeg": .044, "LeftLeg": .032, "RightLeg": .032,
         "LeftFoot": .026, "RightFoot": .026, "LeftToeBase": .02, "RightToeBase": .02, "LowerBack": .075, "Spine": .08,
         "Spine1": .085, "Neck": .03, "Neck1": .03, "LeftShoulder": .03, "RightShoulder": .03, "LeftArm": .027,
         "RightArm": .027, "LeftForeArm": .022, "RightForeArm": .022, "LeftHand": .02, "RightHand": .02}


def read_bvh(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    header, motion = text.split("MOTION", 1)
    lines = [line for line in motion.strip().splitlines() if line.strip()]
    frame_time = float(lines[1].split(":")[1])
    return header, frame_time, np.array([[float(v) for v in line.split()] for line in lines[2:]])


def stride_period(frames, frame_time):
    """Frames in one full stride: the lag where the pose (all channels but the root position) best repeats."""
    pose = frames[:, 6:]
    best, lag_best = None, None
    for lag in range(int(0.45 / frame_time), min(int(1.4 / frame_time), len(frames) - 2)):
        diff = np.abs(pose[lag:] - pose[:-lag])
        diff = np.minimum(diff, 360 - diff).mean()  # angles wrap around
        if best is None or diff < best:
            best, lag_best = diff, lag
    return lag_best


def loop_motion(frames, period, count):
    """Repeat one stride, adding the stride's forward travel each time so the body keeps moving instead of snapping back."""
    cycle = frames[:period]
    travel = frames[period, :3] - frames[0, :3]
    travel[1] = 0  # BVH is Y-up: keep height from drifting
    out = []
    for i in range(count):
        frame = cycle[i % period].copy()
        frame[:3] += travel * (i // period)
        out.append(frame)
    return np.array(out)


def busiest_window(frames, length):
    """Start of the stretch of `length` frames with the most movement, so a long take skips its standing start and end."""
    change = np.abs(np.diff(frames[:, 6:], axis=0))
    change = np.minimum(change, 360 - change).sum(1)
    return int(np.argmax(np.convolve(change, np.ones(length - 1), "valid")))


def resample(frames, frame_time, step_time, count):
    """`count` frames every `step_time` seconds, blended between captured frames, so timing stays even at any frame rate
    (dropping frames to change rate makes a body jerk)."""
    smooth = frames.copy()
    smooth[:, 3:] = np.unwrap(frames[:, 3:], period=360, axis=0)  # so 179 to -179 degrees blends through 180, not 0
    source, times = np.arange(len(frames)) * frame_time, np.arange(count) * step_time
    return np.stack([np.interp(times, source, smooth[:, c]) for c in range(frames.shape[1])], axis=1)


def write_bvh(path, header, frame_time, frames):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "MOTION\n" + f"Frames: {len(frames)}\nFrame Time: {frame_time:.7f}\n")
        f.writelines(" ".join(f"{v:.5f}" for v in frame) + "\n" for frame in frames)


SHIRT = {"LowerBack", "Spine", "Spine1", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm", "LHipJoint", "RHipJoint"}
SHORTS = {"LeftUpLeg", "RightUpLeg"}
SHOES = {"LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"}
# skin, shirt, shorts, shoes, eyes: clothes like a person's help pose models find the body
COLOURS = [(0.78, 0.6, 0.48), (0.18, 0.32, 0.55), (0.08, 0.08, 0.09), (0.9, 0.9, 0.92), (0.02, 0.02, 0.02)]


def plain_material(name, colour, roughness=0.6):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*colour, 1)
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*colour, 1)
    shader.inputs["Roughness"].default_value = roughness
    return material


def build_figure(armature, size):
    """One mesh of capsules, each bound rigidly to its bone, so it moves exactly like the capture. size is the body height in the capture's units."""
    mesh = bpy.data.meshes.new("Figure")
    figure = bpy.data.objects.new("Figure", mesh)
    bpy.context.collection.objects.link(figure)
    bm = bmesh.new()
    groups, parts = {}, {}
    bones = armature.data.bones
    # which way the figure faces in its rest pose: along the toes, flattened
    toe = bones["LeftToeBase"]
    head_bone = bones["Head"]
    up = (head_bone.tail_local - head_bone.head_local).normalized()
    front = toe.tail_local - toe.head_local
    front = (front - up * front.dot(up)).normalized()
    side = up.cross(front)
    for bone in bones:
        radius = RADII.get(bone.name)
        if bone.name == "Head":
            centre = bone.head_local + (bone.tail_local - bone.head_local) * 0.6
            r = size * .062
            made = bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=r, matrix=Matrix.Translation(centre))["verts"]
            parts.update({v: 0 for v in made})
            # a nose and eyes, so the head has a face to find and a direction
            nose = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=8, radius=r * 0.2, matrix=Matrix.Translation(centre + front * r * 0.95 - up * r * 0.1))["verts"]
            parts.update({v: 0 for v in nose})
            eyes = []
            for sign in (1, -1):
                spot = centre + front * r * 0.85 + up * r * 0.22 + side * sign * r * 0.38
                eyes += bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=8, radius=r * 0.13, matrix=Matrix.Translation(spot))["verts"]
            parts.update({v: 4 for v in eyes})
            made = made + nose + eyes
        elif radius:
            vector = bone.tail_local - bone.head_local
            if vector.length < 1e-4:
                continue
            middle = (bone.head_local + bone.tail_local) / 2
            matrix = Matrix.Translation(middle) @ vector.to_track_quat("Z", "Y").to_matrix().to_4x4()
            made = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=size * radius, radius2=size * radius * 0.85,
                                         depth=vector.length + size * radius, matrix=matrix)["verts"]
            parts.update({v: 1 if bone.name in SHIRT else 2 if bone.name in SHORTS else 3 if bone.name in SHOES else 0 for v in made})
        else:
            continue
        groups[bone.name] = made
    bm.verts.index_update()
    for face in bm.faces:
        face.material_index = parts.get(face.verts[0], 0)
    lists = {name: [v.index for v in verts] for name, verts in groups.items()}
    bm.to_mesh(mesh)
    bm.free()
    for name, indices in lists.items():
        figure.vertex_groups.new(name=name).add(indices, 1.0, "REPLACE")
    figure.modifiers.new("Armature", "ARMATURE").object = armature
    figure.parent = armature
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    for i, colour in enumerate(COLOURS):
        mesh.materials.append(plain_material(f"Part{i}", colour))
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh")
    parser.add_argument("out")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--speed", type=float, default=1.0, help="0.5 plays the motion at half speed, to make twice the real frames and speed it back up later")
    parser.add_argument("--depth", action="store_true", help="render distance from the camera (near is white) as a VACE control video")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    header, frame_time, frames = read_bvh(args.bvh)
    count = int(args.seconds * args.fps) + 1
    step_time = args.speed / args.fps
    need = int((count - 1) * step_time / frame_time) + 2
    if len(frames) >= need:
        begin = busiest_window(frames, need)
        motion = frames[begin:begin + need]
        print(f"Using {begin * frame_time:.1f}-{(begin + need) * frame_time:.1f} s of {len(frames) * frame_time:.1f} s", flush=True)
    else:
        period = stride_period(frames, frame_time)
        motion = loop_motion(frames, period, need)
        print(f"Stride: {period * frame_time:.2f} s, looped", flush=True)
    looped = resample(motion, frame_time, step_time, count)
    looped_path = os.path.splitext(args.out)[0] + "_looped.bvh"
    write_bvh(looped_path, header, 1 / args.fps, looped)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = args.fps
    bpy.ops.import_anim.bvh(filepath=looped_path, axis_forward="-Z", axis_up="Y", update_scene_fps=False, update_scene_duration=True)
    armature = bpy.context.selected_objects[0]
    scene.frame_start, scene.frame_end = 1, len(looped)

    # scale to a set height, measured upright in the world on the first frame (bone data may be in another axis order)
    scene.frame_set(scene.frame_start)
    heights = [(armature.matrix_world @ p.head).z for p in armature.pose.bones] + [(armature.matrix_world @ p.tail).z for p in armature.pose.bones]
    tall = max(heights) - min(heights)
    build_figure(armature, tall)
    armature.scale = tuple(v * HEIGHT / tall for v in armature.scale)
    print(f"Capture height {tall:.2f} units, scaled to {HEIGHT} m", flush=True)
    bpy.context.view_layer.update()  # otherwise the new scale isn't in matrix_world yet, and the camera aims at unscaled positions

    # where the hips go, to aim the camera and put the floor under the feet
    hips, lowest = [], 1e9
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        world = armature.matrix_world
        hips.append(world @ armature.pose.bones["Hips"].head)
        lowest = min(lowest, min((world @ p.tail).z for p in armature.pose.bones if "Toe" in p.name or "Foot" in p.name))
    start, end = hips[0], hips[-1]
    if (end - start).length > HEIGHT * 0.5:
        forward = Vector((end.x - start.x, end.y - start.y, 0)).normalized()
        side = Vector((-forward.y, forward.x, 0))
    else:
        # staying in place (dancing, boxing): a still camera in front, from which way the hips face on the first frame
        scene.frame_set(scene.frame_start)
        world = armature.matrix_world
        across = world @ armature.pose.bones["LeftUpLeg"].head - world @ armature.pose.bones["RightUpLeg"].head
        across.z = 0
        side = Matrix.Rotation(math.radians(30), 3, "Z") @ across.normalized().cross(Vector((0, 0, 1)))
        start = end = sum(hips, Vector()) / len(hips)

    bpy.ops.mesh.primitive_plane_add(size=200, location=(start.x, start.y, lowest - HEIGHT * 0.012))
    floor = bpy.context.object
    floor.data.materials.append(plain_material("Floor", (0.2, 0.21, 0.23), 0.9))

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.lens = 40
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    distance = HEIGHT * 3.2
    count, depths = len(hips), []
    for i, frame in enumerate(range(scene.frame_start, scene.frame_end + 1)):
        # follow a straight line at the average speed, so the camera doesn't bob with each step
        along = start.lerp(end, i / max(count - 1, 1))
        target = Vector((along.x, along.y, lowest + HEIGHT * 0.5))
        camera.location = target + side * distance + Vector((0, 0, HEIGHT * 0.05))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        depths.append((camera.location - hips[i]).dot(side))
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(50), 0, math.atan2(side.y, side.x) + math.radians(35))
    bpy.context.collection.objects.link(sun)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.77, 0.8, 1)
    scene.view_settings.view_transform = "Standard"  # plain colours, so the figure stands out from the floor
    scene.world = world
    if args.depth:
        # every surface glows by its distance from the camera, so nothing but shape and distance reaches the control video
        depth = bpy.data.materials.new("Depth")
        depth.use_nodes = True
        nodes, links = depth.node_tree.nodes, depth.node_tree.links
        nodes.clear()
        camera_data, ramp = nodes.new("ShaderNodeCameraData"), nodes.new("ShaderNodeMapRange")
        glow, output = nodes.new("ShaderNodeEmission"), nodes.new("ShaderNodeOutputMaterial")
        # just around the body over the whole clip, so near and far limbs differ and a dancer stepping closer doesn't vanish
        ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = min(depths) - 0.55, max(depths) + 0.75
        ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = 1.0, 0.0
        links.new(camera_data.outputs["View Z Depth"], ramp.inputs["Value"])
        links.new(ramp.outputs["Result"], glow.inputs["Color"])
        links.new(glow.outputs["Emission"], output.inputs["Surface"])
        scene.view_layers[0].material_override = depth
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)

    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x, scene.render.resolution_y = 854, 480
    if hasattr(scene.render.image_settings, "media_type"):  # Blender 5 splits video from image formats
        scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = args.out
    bpy.ops.render.render(animation=True)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    # self-check: a pose that repeats every 10 frames is found, and looping carries the body forward
    t = np.arange(40)
    fake = np.zeros((40, 9))
    fake[:, 0] = t * 0.5
    fake[:, 6] = np.sin(2 * np.pi * t / 10) * 30
    assert stride_period(fake, 0.05) == 10, stride_period(fake, 0.05)
    looped = loop_motion(fake, 10, 25)
    assert looped[20, 0] == 10.0 and looped[20, 6] == fake[0, 6]
    still = np.zeros((30, 7))
    still[12:18, 6] = np.arange(6) * 20
    assert busiest_window(still, 8) in (11, 12), busiest_window(still, 8)
    turn = np.zeros((3, 7))
    turn[:, 0], turn[:, 6] = [0, 1, 2], [170, 179, -172]
    even = resample(turn, 0.1, 0.05, 5)
    assert np.allclose(even[:, 0], [0, .5, 1, 1.5, 2]) and np.isclose(even[3, 6], 183.5), even
    if "--" in sys.argv:
        main()
