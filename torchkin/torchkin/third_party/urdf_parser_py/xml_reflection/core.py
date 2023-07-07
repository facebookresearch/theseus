from torchkin.third_party.urdf_parser_py.xml_reflection.basics import *
import sys
import copy

# @todo Get rid of "import *"
# @todo Make this work with decorators

# Is this reflection or serialization? I think it's serialization...
# Rename?

# Do parent operations after, to allow child to 'override' parameters?
# Need to make sure that duplicate entires do not get into the 'unset*' lists


def reflect(cls, *args, **kwargs):
    """
    Simple wrapper to add XML reflection to an xml_reflection.Object class
    """
    cls.XML_REFL = Reflection(*args, **kwargs)

# Rename 'write_xml' to 'write_xml' to have paired 'load/dump', and make
# 'pre_dump' and 'post_load'?
# When dumping to yaml, include tag name?

# How to incorporate line number and all that jazz?
def on_error_stderr(message):
    """ What to do on an error. This can be changed to raise an exception. """
    sys.stderr.write(message + '\n')
on_error = on_error_stderr


skip_default = False
# defaultIfMatching = True # Not implemeneted yet

# Registering Types
value_types = {}
value_type_prefix = ''


def start_namespace(namespace):
    """
    Basic mechanism to prevent conflicts for string types for URDF and SDF
    @note Does not handle nesting!
    """
    global value_type_prefix
    value_type_prefix = namespace + '.'


def end_namespace():
    global value_type_prefix
    value_type_prefix = ''


def add_type(key, value):
    if isinstance(key, str):
        key = value_type_prefix + key
    assert key not in value_types
    value_types[key] = value


def get_type(cur_type):
    """ Can wrap value types if needed """
    if value_type_prefix and isinstance(cur_type, str):
        # See if it exists in current 'namespace'
        curKey = value_type_prefix + cur_type
        value_type = value_types.get(curKey)
    else:
        value_type = None
    if value_type is None:
        # Try again, in 'global' scope
        value_type = value_types.get(cur_type)
    if value_type is None:
        value_type = make_type(cur_type)
        add_type(cur_type, value_type)
    return value_type


def make_type(cur_type):
    if isinstance(cur_type, ValueType):
        return cur_type
    elif isinstance(cur_type, str):
        if cur_type.startswith('vector'):
            extra = cur_type[6:]
            if extra:
                count = float(extra)
            else:
                count = None
            return VectorType(count)
        else:
            raise Exception("Invalid value type: {}".format(cur_type))
    elif cur_type == list:
        return ListType()
    elif issubclass(cur_type, Object):
        return ObjectType(cur_type)
    elif cur_type in [str, float]:
        return BasicType(cur_type)
    else:
        raise Exception("Invalid type: {}".format(cur_type))


class Path(object):
    def __init__(self, tag, parent=None, suffix="", tree=None):
        self.parent = parent
        self.tag = tag
        self.suffix = suffix
        self.tree = tree # For validating general path (getting true XML path)

    def __str__(self):
        if self.parent is not None:
            return "{}/{}{}".format(self.parent, self.tag, self.suffix)
        else:
            if self.tag is not None and len(self.tag) > 0:
                return "/{}{}".format(self.tag, self.suffix)
            else:
                return self.suffix

class ParseError(Exception):
    def __init__(self, e, path):
        self.e = e
        self.path = path
        message = "ParseError in {}:\n{}".format(self.path, self.e)
        super(ParseError, self).__init__(message)


class ValueType(object):
    """ Primitive value type """

    def from_xml(self, node, path):
        return self.from_string(node.text)

    def write_xml(self, node, value):
        """
        If type has 'write_xml', this function should expect to have it's own
        XML already created i.e., In Axis.to_sdf(self, node), 'node' would be
        the 'axis' element.
        @todo Add function that makes an XML node completely independently?
        """
        node.text = self.to_string(value)

    def equals(self, a, b):
        return a == b


class BasicType(ValueType):
    def __init__(self, cur_type):
        self.type = cur_type

    def to_string(self, value):
        return str(value)

    def from_string(self, value):
        return self.type(value)


class ListType(ValueType):
    def to_string(self, values):
        return ' '.join(values)

    def from_string(self, text):
        return text.split()

    def equals(self, aValues, bValues):
        return len(aValues) == len(bValues) and all(a == b for (a, b) in zip(aValues, bValues))  # noqa


class VectorType(ListType):
    def __init__(self, count=None):
        self.count = count

    def check(self, values):
        if self.count is not None:
            assert len(values) == self.count, "Invalid vector length"

    def to_string(self, values):
        self.check(values)
        raw = list(map(str, values))
        return ListType.to_string(self, raw)

    def from_string(self, text):
        raw = ListType.from_string(self, text)
        self.check(raw)
        return list(map(float, raw))


class RawType(ValueType):
    """
    Simple, raw XML value. Need to bugfix putting this back into a document
    """

    def from_xml(self, node, path):
        return node

    def write_xml(self, node, value):
        # @todo rying to insert an element at root level seems to screw up
        # pretty printing
        children = xml_children(value)
        list(map(node.append, children))
        # Copy attributes
        for (attrib_key, attrib_value) in value.attrib.items():
            node.set(attrib_key, attrib_value)


class SimpleElementType(ValueType):
    """
    Extractor that retrieves data from an element, given a
    specified attribute, casted to value_type.
    """

    def __init__(self, attribute, value_type):
        self.attribute = attribute
        self.value_type = get_type(value_type)

    def from_xml(self, node, path):
        text = node.get(self.attribute)
        return self.value_type.from_string(text)

    def write_xml(self, node, value):
        text = self.value_type.to_string(value)
        node.set(self.attribute, text)


class ObjectType(ValueType):
    def __init__(self, cur_type):
        self.type = cur_type

    def from_xml(self, node, path):
        obj = self.type()
        obj.read_xml(node, path)
        return obj

    def write_xml(self, node, obj):
        obj.write_xml(node)


class FactoryType(ValueType):
    def __init__(self, name, typeMap):
        self.name = name
        self.typeMap = typeMap
        self.nameMap = {}
        for (key, value) in typeMap.items():
            # Reverse lookup
            self.nameMap[value] = key

    def from_xml(self, node, path):
        cur_type = self.typeMap.get(node.tag)
        if cur_type is None:
            raise Exception("Invalid {} tag: {}".format(self.name, node.tag))
        value_type = get_type(cur_type)
        return value_type.from_xml(node, path)

    def get_name(self, obj):
        cur_type = type(obj)
        name = self.nameMap.get(cur_type)
        if name is None:
            raise Exception("Invalid {} type: {}".format(self.name, cur_type))
        return name

    def write_xml(self, node, obj):
        obj.write_xml(node)


class DuckTypedFactory(ValueType):
    def __init__(self, name, typeOrder):
        self.name = name
        assert len(typeOrder) > 0
        self.type_order = typeOrder

    def from_xml(self, node, path):
        error_set = []
        for value_type in self.type_order:
            try:
                return value_type.from_xml(node, path)
            except Exception as e:
                error_set.append((value_type, e))
        # Should have returned, we encountered errors
        out = "Could not perform duck-typed parsing."
        for (value_type, e) in error_set:
            out += "\nValue Type: {}\nException: {}\n".format(value_type, e)
            raise ParseError(Exception(out), path)

    def write_xml(self, node, obj):
        obj.write_xml(node)


class Param(object):
    """ Mirroring Gazebo's SDF api

    @param xml_var: Xml name
            @todo If the value_type is an object with a tag defined in it's
                  reflection, allow it to act as the default tag name?
    @param var: Python class variable name. By default it's the same as the
                XML name
    """

    def __init__(self, xml_var, value_type, required=True, default=None,
                 var=None):
        self.xml_var = xml_var
        if var is None:
            self.var = xml_var
        else:
            self.var = var
        self.type = None
        self.value_type = get_type(value_type)
        self.default = default
        if required:
            assert default is None, "Default does not make sense for a required field"  # noqa
        self.required = required
        self.is_aggregate = False

    def set_default(self, obj):
        if self.required:
            raise Exception("Required {} not set in XML: {}".format(self.type, self.xml_var))  # noqa
        elif not skip_default:
            setattr(obj, self.var, self.default)


class Attribute(Param):
    def __init__(self, xml_var, value_type, required=True, default=None,
                 var=None):
        Param.__init__(self, xml_var, value_type, required, default, var)
        self.type = 'attribute'

    def set_from_string(self, obj, value):
        """ Node is the parent node in this case """
        # Duplicate attributes cannot occur at this point
        setattr(obj, self.var, self.value_type.from_string(value))

    def get_value(self, obj):
        return getattr(obj, self.var)

    def add_to_xml(self, obj, node):
        value = getattr(obj, self.var)
        # Do not set with default value if value is None
        if value is None:
            if self.required:
                raise Exception("Required attribute not set in object: {}".format(self.var))  # noqa
            elif not skip_default:
                value = self.default
        # Allow value type to handle None?
        if value is not None:
            node.set(self.xml_var, self.value_type.to_string(value))

# Add option if this requires a header?
# Like <joints> <joint/> .... </joints> ???
# Not really... This would be a specific list type, not really aggregate


class Element(Param):
    def __init__(self, xml_var, value_type, required=True, default=None,
                 var=None, is_raw=False):
        Param.__init__(self, xml_var, value_type, required, default, var)
        self.type = 'element'
        self.is_raw = is_raw

    def set_from_xml(self, obj, node, path):
        value = self.value_type.from_xml(node, path)
        setattr(obj, self.var, value)

    def add_to_xml(self, obj, parent):
        value = getattr(obj, self.xml_var)
        if value is None:
            if self.required:
                raise Exception("Required element not defined in object: {}".format(self.var))  # noqa
            elif not skip_default:
                value = self.default
        if value is not None:
            self.add_scalar_to_xml(parent, value)

    def add_scalar_to_xml(self, parent, value):
        if self.is_raw:
            node = parent
        else:
            node = node_add(parent, self.xml_var)
        self.value_type.write_xml(node, value)


class AggregateElement(Element):
    def __init__(self, xml_var, value_type, var=None, is_raw=False):
        if var is None:
            var = xml_var + 's'
        Element.__init__(self, xml_var, value_type, required=False, var=var,
                         is_raw=is_raw)
        self.is_aggregate = True

    def add_from_xml(self, obj, node, path):
        value = self.value_type.from_xml(node, path)
        obj.add_aggregate(self.xml_var, value)

    def set_default(self, obj):
        pass


class Info:
    """ Small container for keeping track of what's been consumed """

    def __init__(self, node):
        self.attributes = list(node.attrib.keys())
        self.children = xml_children(node)


class Reflection(object):
    def __init__(self, params=[], parent_cls=None, tag=None):
        """ Construct a XML reflection thing
        @param parent_cls: Parent class, to use it's reflection as well.
        @param tag: Only necessary if you intend to use Object.write_xml_doc()
                This does not override the name supplied in the reflection
                definition thing.
        """
        if parent_cls is not None:
            self.parent = parent_cls.XML_REFL
        else:
            self.parent = None
        self.tag = tag

        # Laziness for now
        attributes = []
        elements = []
        for param in params:
            if isinstance(param, Element):
                elements.append(param)
            else:
                attributes.append(param)

        self.vars = []
        self.paramMap = {}

        self.attributes = attributes
        self.attribute_map = {}
        self.required_attribute_names = []
        for attribute in attributes:
            self.attribute_map[attribute.xml_var] = attribute
            self.paramMap[attribute.xml_var] = attribute
            self.vars.append(attribute.var)
            if attribute.required:
                self.required_attribute_names.append(attribute.xml_var)

        self.elements = []
        self.element_map = {}
        self.required_element_names = []
        self.aggregates = []
        self.scalars = []
        self.scalarNames = []
        for element in elements:
            self.element_map[element.xml_var] = element
            self.paramMap[element.xml_var] = element
            self.vars.append(element.var)
            if element.required:
                self.required_element_names.append(element.xml_var)
            if element.is_aggregate:
                self.aggregates.append(element)
            else:
                self.scalars.append(element)
                self.scalarNames.append(element.xml_var)

    def set_from_xml(self, obj, node, path, info=None):
        is_final = False
        if info is None:
            is_final = True
            info = Info(node)

        if self.parent:
            path = self.parent.set_from_xml(obj, node, path, info)

        # Make this a map instead? Faster access? {name: isSet} ?
        unset_attributes = list(self.attribute_map.keys())
        unset_scalars = copy.copy(self.scalarNames)

        def get_attr_path(attribute):
            attr_path = copy.copy(path)
            attr_path.suffix += '[@{}]'.format(attribute.xml_var)
            return attr_path

        def get_element_path(element):
            element_path = Path(element.xml_var, parent = path)
            # Add an index (allow this to be overriden)
            if element.is_aggregate:
                values = obj.get_aggregate_list(element.xml_var)
                index = 1 + len(values) # 1-based indexing for W3C XPath
                element_path.suffix = "[{}]".format(index)
            return element_path

        id_var = "name"
        # Better method? Queues?
        for xml_var in copy.copy(info.attributes):
            attribute = self.attribute_map.get(xml_var)
            if attribute is not None:
                value = node.attrib[xml_var]
                attr_path = get_attr_path(attribute)
                try:
                    attribute.set_from_string(obj, value)
                    if attribute.xml_var == id_var:
                        # Add id_var suffix to current path (do not copy so it propagates)
                        path.suffix = "[@{}='{}']".format(id_var, attribute.get_value(obj))
                except ParseError:
                    raise
                except Exception as e:
                    raise ParseError(e, attr_path)
                unset_attributes.remove(xml_var)
                info.attributes.remove(xml_var)

        # Parse unconsumed nodes
        for child in copy.copy(info.children):
            tag = child.tag
            element = self.element_map.get(tag)
            if element is not None:
                # Name will have been set
                element_path = get_element_path(element)
                if element.is_aggregate:
                    element.add_from_xml(obj, child, element_path)
                else:
                    if tag in unset_scalars:
                        element.set_from_xml(obj, child, element_path)
                        unset_scalars.remove(tag)
                    else:
                        on_error("Scalar element defined multiple times: {}".format(tag))  # noqa
                info.children.remove(child)

        # For unset attributes and scalar elements, we should not pass the attribute
        # or element path, as those paths will implicitly not exist.
        # If we do supply it, then the user would need to manually prune the XPath to try
        # and find where the problematic parent element.
        for attribute in map(self.attribute_map.get, unset_attributes):
            try:
                attribute.set_default(obj)
            except ParseError:
                raise
            except Exception as e:
                raise ParseError(e, path) # get_attr_path(attribute.xml_var)

        for element in map(self.element_map.get, unset_scalars):
            try:
                element.set_default(obj)
            except ParseError:
                raise
            except Exception as e:
                raise ParseError(e, path) # get_element_path(element)

        if is_final:
            for xml_var in info.attributes:
                on_error('Unknown attribute "{}" in {}'.format(xml_var, path))
            for node in info.children:
                on_error('Unknown tag "{}" in {}'.format(node.tag, path))
        # Allow children parsers to adopt this current path (if modified with id_var)
        return path

    def add_to_xml(self, obj, node):
        if self.parent:
            self.parent.add_to_xml(obj, node)
        for attribute in self.attributes:
            attribute.add_to_xml(obj, node)
        for element in self.scalars:
            element.add_to_xml(obj, node)
        # Now add in aggregates
        if self.aggregates:
            obj.add_aggregates_to_xml(node)


class Object(YamlReflection):
    """ Raw python object for yaml / xml representation """
    XML_REFL = None

    def get_refl_vars(self):
        return self.XML_REFL.vars

    def check_valid(self):
        pass

    def pre_write_xml(self):
        """ If anything needs to be converted prior to dumping to xml
        i.e., getting the names of objects and such """
        pass

    def write_xml(self, node):
        """ Adds contents directly to XML node """
        self.check_valid()
        self.pre_write_xml()
        self.XML_REFL.add_to_xml(self, node)

    def to_xml(self):
        """ Creates an overarching tag and adds its contents to the node """
        tag = self.XML_REFL.tag
        assert tag is not None, "Must define 'tag' in reflection to use this function"  # noqa
        doc = etree.Element(tag)
        self.write_xml(doc)
        return doc

    def to_xml_string(self, addHeader=True):
        return xml_string(self.to_xml(), addHeader)

    def post_read_xml(self):
        pass

    def read_xml(self, node, path):
        self.XML_REFL.set_from_xml(self, node, path)
        self.post_read_xml()
        try:
            self.check_valid()
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(e, path)

    @classmethod
    def from_xml(cls, node, path):
        cur_type = get_type(cls)
        return cur_type.from_xml(node, path)

    @classmethod
    def from_xml_string(cls, xml_string):
        node = etree.fromstring(xml_string)
        path = Path(cls.XML_REFL.tag, tree=etree.ElementTree(node))
        return cls.from_xml(node, path)

    @classmethod
    def from_xml_file(cls, file_path):
        xml_string = open(file_path, 'r').read()
        return cls.from_xml_string(xml_string)

    # Confusing distinction between loading code in object and reflection
    # registry thing...

    def get_aggregate_list(self, xml_var):
        var = self.XML_REFL.paramMap[xml_var].var
        values = getattr(self, var)
        assert isinstance(values, list)
        return values

    def aggregate_init(self):
        """ Must be called in constructor! """
        self.aggregate_order = []
        # Store this info in the loaded object??? Nah
        self.aggregate_type = {}

    def add_aggregate(self, xml_var, obj):
        """ NOTE: One must keep careful track of aggregate types for this system.
        Can use 'lump_aggregates()' before writing if you don't care. """
        self.get_aggregate_list(xml_var).append(obj)
        self.aggregate_order.append(obj)
        self.aggregate_type[obj] = xml_var

    def add_aggregates_to_xml(self, node):
        for value in self.aggregate_order:
            typeName = self.aggregate_type[value]
            element = self.XML_REFL.element_map[typeName]
            element.add_scalar_to_xml(node, value)

    def remove_aggregate(self, obj):
        self.aggregate_order.remove(obj)
        xml_var = self.aggregate_type[obj]
        del self.aggregate_type[obj]
        self.get_aggregate_list(xml_var).remove(obj)

    def lump_aggregates(self):
        """ Put all aggregate types together, just because """
        self.aggregate_init()
        for param in self.XML_REFL.aggregates:
            for obj in self.get_aggregate_list(param.xml_var):
                self.add_aggregate(param.var, obj)

    """ Compatibility """

    def parse(self, xml_string):
        node = etree.fromstring(xml_string)
        path = Path(self.XML_REFL.tag, tree=etree.ElementTree(node))
        self.read_xml(node, path)
        return self


# Really common types
# Better name: element_with_name? Attributed element?
add_type('element_name', SimpleElementType('name', str))
add_type('element_value', SimpleElementType('value', float))

# Add in common vector types so they aren't absorbed into the namespaces
get_type('vector3')
get_type('vector4')
get_type('vector6')