"""Image smoke checks against an already running GLM-5.3-Flash server.

GLM53_TEST_URL=http://127.0.0.1:11435 python test/manual/test_glm53_flash_vision.py
Run once with the launcher's default MTP and once with GLM53_MTP_STEPS=0.
"""

import base64
import io
import os
import re
import unittest

import requests
from PIL import Image

from sglang.test.test_utils import CustomTestCase


class TestGLM53FlashVision(CustomTestCase):
    base_url = os.environ.get("GLM53_TEST_URL", "http://127.0.0.1:11435").rstrip("/")
    model = os.environ.get("GLM53_TEST_MODEL", "glm53-flash-nvfp4")

    def test_image_colors(self):
        # Identical text with different pixels catches image inputs being ignored,
        # including stale multimodal prefix-cache reuse on the second request.
        prompt = "What is the color of this image? Answer with one color word."
        for api in ("openai", "anthropic"):
            for color, size in (("red", 224), ("blue", 224), ("red", 1120)):
                with self.subTest(api=api, color=color, size=size):
                    buffer = io.BytesIO()
                    Image.new("RGB", (size, size), color=color).save(
                        buffer, format="PNG"
                    )
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    if api == "openai":
                        path = "/v1/chat/completions"
                        content = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ]
                        extra = {"reasoning_effort": "low"}
                    else:
                        path = "/v1/messages"
                        content = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ]
                        extra = {"output_config": {"effort": "low"}}
                    response = requests.post(
                        self.base_url + path,
                        headers={"anthropic-version": "2023-06-01"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": 2048,
                            "temperature": 0,
                            **extra,
                        },
                        timeout=300,
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    if api == "openai":
                        answer = result["choices"][0]["message"].get("content") or ""
                    else:
                        answer = " ".join(
                            block["text"]
                            for block in result["content"]
                            if block["type"] == "text"
                        )
                    print(f"{api} {color} ({size}x{size}): {answer!r}", flush=True)
                    self.assertRegex(answer.lower(), rf"\b{color}\b", result)
                    other_color = "blue" if color == "red" else "red"
                    self.assertIsNone(
                        re.search(rf"\b{other_color}\b", answer.lower()), result
                    )


if __name__ == "__main__":
    unittest.main()
