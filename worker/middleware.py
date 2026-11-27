from starlette.responses import JSONResponse


class RequestBodyLimitMiddleware:
    """Reject HTTP requests exceeding a configured body-size limit."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body = bytearray()

        while True:
            message = await receive()

            if message["type"] == "http.disconnect":
                return

            chunk = message.get("body", b"")

            if len(body) + len(chunk) > self.max_body_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": "Request body exceeds the maximum allowed size."
                    },
                )

                await response(scope, receive, send)
                return

            body.extend(chunk)

            if not message.get("more_body", False):
                break

        body_sent = False

        async def receive_with_body():
            nonlocal body_sent

            if not body_sent:
                body_sent = True

                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }

            return await receive()

        await self.app(scope, receive_with_body, send)
