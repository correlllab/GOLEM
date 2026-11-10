"""Stand in for the planner worker in test_planner.py's client tests.

Each runs in a spawned process in place of planner_process._planner_worker_main, with the
same arguments and message protocol, and answers a request with the request's goal as a
list. Only the standard library is imported here, so a stand-in starts in a fraction of a
second.
"""
import os
import time


def _answer(request):
    """Return the result message for request: its goal as a list."""
    return {'id': request['id'], 'type': 'result', 'result': [float(x) for x in request['goal']]}


def _serve(request_queue, response_queue, respond):
    """Call respond(request, response_queue) per request until a shutdown request."""
    while True:
        request = request_queue.get()
        if request.get('command') == 'shutdown':
            return
        respond(request, response_queue)


def echo(init_args, request_queue, response_queue):
    """Report ready at once; answer every request."""
    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue,
           lambda request, responses: responses.put(_answer(request)))


def ready_after_a_second(init_args, request_queue, response_queue):
    """Report ready after 1 s, as a slow model build would; then answer every request."""
    time.sleep(1.0)
    echo(init_args, request_queue, response_queue)


def stale_answer_first(init_args, request_queue, response_queue):
    """Before each answer, send one for the request before it, as a late reply would."""
    def respond(request, responses):
        """Send a stale reply, then the answer."""
        responses.put({'id': request['id'] - 1, 'type': 'result', 'result': 'stale'})
        responses.put(_answer(request))

    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue, respond)


def dies_at_start(init_args, request_queue, response_queue):
    """Exit with code 3 before reporting ready."""
    os._exit(3)


def never_answers(init_args, request_queue, response_queue):
    """Report ready at once; take requests without answering them until told to shut down."""
    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue, lambda request, responses: None)


def echo_after_a_while(init_args, request_queue, response_queue):
    """Report ready at once; answer each request 0.2 s after taking it."""
    def respond(request, responses):
        """Wait 0.2 s, then answer."""
        time.sleep(0.2)
        responses.put(_answer(request))

    response_queue.put({'type': 'ready'})
    _serve(request_queue, response_queue, respond)
