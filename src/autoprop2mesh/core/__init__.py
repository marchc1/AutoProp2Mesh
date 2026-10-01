"""Blender-independent core: Source engine file formats, the GMod virtual
filesystem, and the AdvDupe2 / Prop2Mesh serializers.

Nothing in this package imports bpy, so it can be unit tested with a plain
Python interpreter (numpy is required for model and texture decoding).
"""
