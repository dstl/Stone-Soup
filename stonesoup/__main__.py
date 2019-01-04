import argparse

parser = argparse.ArgumentParser(
    prog=__package__,
    description="Stone Soup command line interface and webui launcher",
)
subparsers = parser.add_subparsers(title="Commands", dest='command')

# Web UI
webui_parser = subparsers.add_parser(
    'webui', help="Run Stone Soup's web based UI")
webui_parser.add_argument(
    '--host', default="localhost",
    help="host in which to run web ui")
webui_parser.add_argument(
    '--port', default=5000, type=int,
    help="port in which to run web ui")

args = parser.parse_args()

if args.command == 'webui':
    import logging

    from stonesoup.webui import logger
    from stonesoup.webui.config import socketio  # noqa:F401
    from stonesoup.webui.plot import dash_app

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    logger.info(
        "Starting stone soup web ui on http://{0.host}:{0.port}/".format(args))
    dash_app.run_server(host=args.host, port=args.port)
