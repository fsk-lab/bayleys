import abc


def non_abstract_subclasses(cls) -> set:
    """
    Returns a set of all non-abstract subclasses of a given class.

    Args:
        cls: The class to find non-abstract subclasses for.

    Returns:
        set: A set of non-abstract subclasses of the given class.
    """
    subclasses = set()

    for subclass in cls.__subclasses__():

        if abc.ABC not in subclass.__bases__:
            subclasses.add(subclass)

        subclasses.update(non_abstract_subclasses(subclass))

    return subclasses