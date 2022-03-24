import theseus as th


def mypy_operations_check():

    for point_cls in [th.Point2, th.Point3]:

        x = point_cls()
        y = point_cls()

        z: point_cls = x + y  # noqa: F841
        z: point_cls = x - y  # noqa: F841
        z: point_cls = x * y  # noqa: F841
        z: point_cls = x / y  # noqa: F841
