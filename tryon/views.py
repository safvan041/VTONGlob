"""Django view handling the jewelry try‑on workflow.

The single endpoint ``/tryon/`` renders a form (GET) and processes the
uploaded files (POST). It saves the uploads to ``MEDIA_ROOT`` and invokes
the appropriate engine function based on the selected jewelry type.
"""

import os
import uuid
from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, FileResponse
from django.views.decorators.http import require_http_methods

# Registry mapping the dropdown value to the engine callable.
from .jewellery.registry import REGISTRY

# Ensure the media folder exists.
MEDIA_ROOT = getattr(settings, "MEDIA_ROOT", Path(__file__).resolve().parent / "media")
os.makedirs(MEDIA_ROOT, exist_ok=True)


@require_http_methods(["GET", "POST"])
def tryon_view(request):
    if request.method == "GET":
        return render(request, "tryon/form.html")

    # POST handling
    uploaded_face = request.FILES.get("face_image")
    uploaded_jewel = request.FILES.get("jewellery_image")
    jewellery_type = request.POST.get("jewellery_type")
    if not (uploaded_face and uploaded_jewel and jewellery_type):
        return HttpResponseBadRequest("Missing required fields.")

    if jewellery_type not in REGISTRY:
        return HttpResponseBadRequest("Invalid jewellery type.")

    # Save uploaded files with a unique name to avoid collisions.
    uid = uuid.uuid4().hex
    face_path = os.path.join(MEDIA_ROOT, f"{uid}_face{Path(uploaded_face.name).suffix}")
    jewel_path = os.path.join(
        MEDIA_ROOT, f"{uid}_jewel{Path(uploaded_jewel.name).suffix}"
    )
    output_path = os.path.join(MEDIA_ROOT, f"{uid}_result.png")

    with open(face_path, "wb") as f:
        for chunk in uploaded_face.chunks():
            f.write(chunk)
    with open(jewel_path, "wb") as f:
        for chunk in uploaded_jewel.chunks():
            f.write(chunk)

    # Call the appropriate engine.
    engine_func = REGISTRY[jewellery_type]
    try:
        engine_func(face_path, jewel_path, output_path)
    except Exception as exc:  # pragma: no cover – shown to user for debugging
        return HttpResponseBadRequest(f"Processing error: {exc}")

    # Serve the resulting image directly (could also redirect to a URL).
    return FileResponse(open(output_path, "rb"), content_type="image/png")
