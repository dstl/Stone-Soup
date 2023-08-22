from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from reactive_isr_core.data import Algorithm, ProcessingStatistics, Storage, Node, \
    AvailableAlgorithms, Availability, Edge, TraversalTime, NetworkTopology, AssetList, Asset, \
    AssetDescription, SensorType, AssetStatus, GeoLocation, RFI, \
    TaskType, GeoRegion, PriorityOverTime, ThresholdOverTime


def setup_network(sensor_location, start_time):
    algorithms = [
        Algorithm(
            cost=1,
            prob_detection=0.6,
            false_alarm_density=0.1,
            name="Algorithm1",
            processing_statistics=ProcessingStatistics(
                mu=1,
                sigma=0.1,
                lower_truncation=0
            )
        ),
        Algorithm(
            cost=2,
            prob_detection=0.75,
            false_alarm_density=0.05,
            name="Algorithm2",
            processing_statistics=ProcessingStatistics(
                mu=3,
                sigma=0.1,
                lower_truncation=0
            )
        ),
        Algorithm(
            cost=3,
            prob_detection=0.9,
            false_alarm_density=0.01,
            name="Algorithm3",
            processing_statistics=ProcessingStatistics(
                mu=5,
                sigma=0.1,
                lower_truncation=0
            )
        )
    ]

    dummy_storage = Storage(
        capacity=1,
        contents=[]
    )

    sensor_node = Node(
        id=uuid4(),
        processing_capability=AvailableAlgorithms(
            algorithms=[algorithms[0]]
        ),
        total_task_capacity=dict(),
        availability=Availability.AVAILABLE,
        storage=dummy_storage,
        peers=[]
    )

    fob_node = Node(
        id=uuid4(),
        processing_capability=AvailableAlgorithms(
            algorithms=[algorithms[1]]
        ),
        total_task_capacity=dict(),
        availability=Availability.AVAILABLE,
        storage=dummy_storage,
        peers=[]
    )

    cic_node = Node(
        id=uuid4(),
        processing_capability=AvailableAlgorithms(
            algorithms=[algorithms[2]]
        ),
        total_task_capacity=dict(),
        availability=Availability.AVAILABLE,
        storage=dummy_storage,
        peers=[]
    )

    nodes = [sensor_node, fob_node, cic_node]

    edges = [
        Edge(
            id=uuid4(),
            source_node=sensor_node.id,
            target_node=fob_node.id,
            traversal_time=TraversalTime(
                mu=1,
                sigma=0.1,
                lower_truncation=0
            )
        ),
        Edge(
            id=uuid4(),
            source_node=fob_node.id,
            target_node=cic_node.id,
            traversal_time=TraversalTime(
                mu=3,
                sigma=0.1,
                lower_truncation=0
            )
        ),
        Edge(
            id=uuid4(),
            source_node=sensor_node.id,
            target_node=cic_node.id,
            traversal_time=TraversalTime(
                mu=4,
                sigma=0.2,
                lower_truncation=0
            )
        ),
    ]

    sensor_node.peers = [fob_node.id]
    fob_node.peers = [sensor_node.id, cic_node.id]
    cic_node.peers = [fob_node.id]

    network_topology = NetworkTopology(
        nodes=nodes,
        edges=edges
    )

    assets = AssetList(
        assets=[
            Asset(
                asset_description=AssetDescription(
                    id=sensor_node.id,
                    name="Sensor",
                    sensor_types=[SensorType.AERIAL_V_CAMERA],
                    response_timeout=1,
                    fov_radius=30,
                ),
                asset_status=AssetStatus(
                    time=start_time,
                    id=sensor_node.id,
                    location=GeoLocation(
                        latitude=sensor_location[1],
                        longitude=sensor_location[0],
                        altitude=sensor_location[2]
                    ),
                    availability=Availability.AVAILABLE
                ),
                target_detections=[]
            ),
        ]
    )

    return network_topology, assets


def setup_rfis(start_time, num_rois, time_varying):
    roi1 = GeoRegion(corners=[
        GeoLocation(
            longitude=-3.3,
            latitude=51.1,
            altitude=0),
        GeoLocation(
            longitude=-2.9,
            latitude=51.5,
            altitude=0)]
    )
    roi2 = GeoRegion(corners=[
        GeoLocation(
            longitude=-2.4,
            latitude=52.1,
            altitude=0),
        GeoLocation(
            longitude=-2,
            latitude=52.5,
            altitude=0)]
    )
    rois=[roi1]
    if num_rois > 1:
        rois.append(roi2)
    priority = [5, 5]
    if time_varying:
        priority = [5, 0]
    rfi = RFI(id=uuid4(),
              task_type=TaskType.COUNT,
              region_of_interest=rois,
              start_time=datetime.now(),
              end_time=datetime.now(),
              priority_over_time=PriorityOverTime(
                  timescale=[start_time, start_time+timedelta(seconds=400)],
                  priority=priority),
              targets=[],
              threshold_over_time=ThresholdOverTime(timescale=[start_time],
                                                    threshold=[.01]))
    return [rfi]


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def compute_ellipse(cov, pos, nstd=1, **kwargs):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)
    return ellip.get_path()


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip
