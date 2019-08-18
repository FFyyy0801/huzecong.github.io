---
layout: post
title:  "Inheritance for Python Namedtuples"
date:   2019-08-10 15:05:00 -0400
categories: tech
locale: en
---

>  **tl;dr:** Inheritance for the Python built-in namedtuple does not work as we expect. This blog post demonstrates how to create a custom namedtuple class that supports meaningful inheritance, and more.

I've always under-appreciated the Python [`collections.namedtuple`](https://docs.python.org/3/library/collections.html#collections.namedtuple) class. For those who are unfamiliar, a namedtuple is a fancier tuple, whose elements can also be accessed as attributes:

```python
from collections import namedtuple

Point = namedtuple('Point', ('x', 'y'))
p = Point(1, 2)
print(p.x, p.y)  # 1 2
```

This allows using meaningful names for the elements, rather than having to remember what are stored under each index.

What I don't like about it, however, is the ugly syntax: attribute names are stored as strings, the class name is repeated, and most importantly, refactoring is error-prone, even within powerful IDEs. You can rename class attributes and all the references easily in PyCharm, but you can't do that for namedtuples. What I wanted was a syntax like that of the C/C++ `struct`, with a default constructor to assign values to each field.

Luckily, this changed in Python 3.6, with the implementation of [PEP 526](https://www.python.org/dev/peps/pep-0526/). This version provides [`typing.NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple), a typed version of namedtuple with a brand new syntax. Instead of the example above, now you can write:

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: int
    y: int = 0

p = Point(1)
print(p.x, p.y)  # 1 0
```

This snippet works in exactly the same way, but adds type annotations for each field, and also supports default values (but fields with default values have to follow those without, just as in a function declaration). The syntax is also much more natural (to a former C++ user, at least). But there is still something we can't do: inheritance.

If you ever tried to inherit a namedtuple, you will find that it doesn't work as you expect. As illustrated in [this StackOverflow question](https://stackoverflow.com/questions/42385916/inheriting-from-a-namedtuple-base-class-python), the new attributes added in the subclass doesn't show up, and you'd have to manually override the constructor, which is kind of against the intention of using namedtuples in the first place.

Now, you may think, let's just hack into the internals and somehow make inheritance work. If you were ever in the mood to peek under the hood of this `namedtuple` class, you'd find that it's surprisingly complicated for what seemed like a small and easy piece of functionality. But don't be afraid, the logic is actually pretty straightforward — it just involves some details of Python's internal data model.

Before we begin, let's summarize what we want to achieve through this blog post:

- Make inheritance work for `typing.NamedTuple` as we expect.
- Also allow multiple inheritance, if there are no overlaps in field names among the base classes.
- Remove the constraint on ordering for fields with default values.

## Instance, Class, and Metaclass

Before diving into the actual code, let's get a couple of concepts clear. We need to know what metaclasses are, and how a class is created, before we can customize that behavior.

If you're not familiar with metaclasses, I recommend reading [this wonderful article](https://blog.ionelmc.ro/2015/02/09/understanding-python-metaclasses/), which gives a comprehensive explanation of the entire topic. But here, I will try to briefly explain the concepts that will be useful for our goals.

#### Class Instance and the `__new__` method

We're all familiar with **class**es. An **instance** of a class is what you'd get after calling the class constructor.

You might think the Python class constructor is `__init__`, but that's not the whole story. When you construct an instance, the `__new__` method is first called with the same arguments you pass to `__init__`. `__new__` is responsible for the actually creating an instance of the class, and that instance is then passed into `__init__` as the `self` argument.

Note that `__new__` is considered a [class method](https://docs.python.org/3/library/functions.html#classmethod) (because the instance is not even created at the point of call), so its first argument is `cls` instead of `self`. For most classes, the `__new__` method just calls the super class `__new__`, which all traces back to `object.__new__(cls)`.

There are special cases though — you can return stuff that is not an instance of type `cls` (or any of its subclasses), in which case, the `__init__` method will not be called. A common use case for this is to entirely disable the behaviors of a class:

```python
class ProgressBar:  # wrap around an iterable to print a progress bar to terminal
    def __new__(cls, iterable, enable=True):
        if not enable:
            return iterable  # progress bar disabled; don't wrap the iterable
        return super().__new__(cls)

    def __init__(self, iterable, enable=True):
        # `enable` must be `True`
```

#### Metaclass

The `type` built-in function shows the type of objects, *e.g.*,

```python
type(2)  # int
type(3.14)  # float
type("wow")  # str
type([1, 2, 3, 4])  # list
type(MissileWarningSystem(test_run=False))  # <class 'MissileWarningSystem'>
```

But what is the type of a class? Turns out, the type of a class is what we call a **metaclass**, and the default metaclass (and the base for all metaclasses) is `type` itself. This reveals a new level of hierarchy[^1] to us:

- An instance is an instance of a class. The base for all classes is `object`.
- A class is an instance of a metaclass. The base for all metaclasses is `type`.

Just as classes control the behavior of instances, metaclasses control the behavior of classes. When a class is created, the metaclass' `__new__` method is called, and then its `__init__` method. What's different to classes is that you don't get to customize the arguments received, it's always like this:

```python
class Metaclass(type):
    def __new__(mcs, typename, bases, namespace): ...
```

- `mcs` is the metaclass instance, in this case, `Metaclass` or its potential sub-metaclasses (yes, inheritance works here).
- `typename` is a `str` storing the name of the class to create.
- `bases` is a tuple of classes, containing the base classes of the class to create. This is what's in the brackets following the class name on the first line.
- `namespace` contains all the class-level attributes, including methods and class attributes.

Since `type` is the default metaclass, we can use the same set of arguments with the `type` constructor to programmatically create a new class:

```python
MyClass = type("MyClass", (object,), {
    "__init__": lambda self, x: setattr(self, 'x', x),
    "foo": lambda self: print(self.x),
})
```

which is equivalent to the canonical class definition syntax:

```python
class MyClass(object):
    def __init__(self, x):
        self.x = x
    def foo(self):
        print(self.x)
```

[^1]: There's actually another level called the meta-metaclass, but that's rarely useful and I've never seen any practical usages.

## The `NamedTuple` Class

Now that we're equipped with the adequate knowledge, the first thing to do is look at how `NamedTuple` is implemented:

```python
def _make_nmtuple(name, types):
    msg = "NamedTuple('Name', [(f0, t0), (f1, t1), ...]); each t must be a type"
    types = [(n, _type_check(t, msg)) for n, t in types]
    nm_tpl = collections.namedtuple(name, [n for n, t in types])
    # Prior to PEP 526, only _field_types attribute was assigned.
    # Now, both __annotations__ and _field_types are used to maintain compatibility.
    nm_tpl.__annotations__ = nm_tpl._field_types = collections.OrderedDict(types)
    try:
        nm_tpl.__module__ = sys._getframe(2).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        pass
    return nm_tpl

class NamedTuple(metaclass=NamedTupleMeta):
    _root = True

    def __new__(self, typename, fields=None, **kwargs):
        if fields is None:
            fields = kwargs.items()
        elif kwargs:
            raise TypeError("Either list of fields or keywords"
                            " can be provided to NamedTuple, not both")
        return _make_nmtuple(typename, fields)
```

The `__new__` method here is actually not of our interest — it's just here to provide an interface similar to `namedtuple`. The `_make_nmtuple` function that's called from `__new__` is a utility function that internally constructs a `collections.namedtuple` and adds type annotations to it. We note that what's returned from `__new__` is not an instance of `NamedTuple`.

We notice that `NamedTuple` has a metaclass called `NamedTupleMeta`. The `_root` attribute here is important for the metaclass, and we'll talk more of it later.

## The `NamedTupleMeta` Metaclass

Now let's take a look at the metaclass code:

```python
class NamedTupleMeta(type):

    def __new__(cls, typename, bases, ns):
        if ns.get('_root', False):
            return super().__new__(cls, typename, bases, ns)
        types = ns.get('__annotations__', {})
        nm_tpl = _make_nmtuple(typename, types.items())
        defaults = []
        defaults_dict = {}
        for field_name in types:
            if field_name in ns:
                default_value = ns[field_name]
                defaults.append(default_value)
                defaults_dict[field_name] = default_value
            elif defaults:
                raise TypeError("Non-default namedtuple field {field_name} cannot "
                                "follow default field(s) {default_names}"
                                .format(field_name=field_name,
                                        default_names=', '.join(defaults_dict.keys())))
        nm_tpl.__new__.__annotations__ = collections.OrderedDict(types)
        nm_tpl.__new__.__defaults__ = tuple(defaults)
        nm_tpl._field_defaults = defaults_dict
        # update from user namespace without overriding special namedtuple attributes
        for key in ns:
            if key in _prohibited:
                raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
            elif key not in _special and key not in nm_tpl._fields:
                setattr(nm_tpl, key, ns[key])
        return nm_tpl
```

Now we know why there's a `_root` attribute in `NamedTuple`. The `__new__` method of `NamedTupleMeta` is also called when `NamedTuple` is created, but we can't create a `collections.namedtuple` for that. Thus, we check whether this special `_root` attribute exists, and skips the following procedure if it does.

When a subclass of `NamedTuple` is created, the `__new__` method is also called, but this time the rest of the procedure is also executed. A couple of things happen:

- Obtain the list of fields in the namedtuple definition. Since we provide an annotation for each field, they're stored as a dictionary in the `__annotations__` special attribute of the class.
- Create a namedtuple class using `_make_nmtuple`. Note that the returned namedtuple class does not support default values[^2] or contain type annotations for the `__init__` method.
- Gather default values from `ns` (namespace) and set annotations and default argument values for the `__new__` method of the namedtuple class.
- Add other attributes and methods to the created namedtuple class, so additional methods you defined in the `NamedTuple` subclass can also be called from the returned namedtuple class.

[^2]: This is true for Python 3.6 and lower. Starting from Python 3.7, `collections.namedtuple` supports an optional `default` argument.

## Inheritance with a Single Base Class

Let's first think about what we're trying to accomplish by inheritance:

- Automatically generate a constructor that sets all fields, including those from the base class.
- Access methods, attributes, and properties from the base class.
- Behave correctly in `isinstance` and `issubclass` checks.

If we don't care about the latter two, the solution is pretty straightforward: we just gather the fields defined in the derived and base classes, and ask `NamedTupleMeta` to create a `NamedTuple` based on these fields.

Let's make a first attempt at implementing this. Out of personal preference, I'm going to call our enhanced namedtuple `Options`.

```python
class OptionsMeta(typing.NamedTupleMeta):
    def __new__(mcs, typename, bases, namespace):
        if namespace.get('_root', False):
            # The created class is `Options`, skip.
            return super().__new__(mcs, typename, bases, namespace)

        # Gather fields from annotations of current class and base class.
        fields = collections.OrderedDict()
        cur_fields = namespace.get('__annotations__', {})
        # We only deal with single inheritance for now.
        assert len(bases) == 1
        base = bases[0]
        if hasattr(base, '_fields'):
            # Base class is a concrete namedtuple.
            for name in base._fields:
                # Make sure not to overwrite redefined fields.
                if name not in cur_fields:
                    fields[name] = base.__annotations__[name]
                    if name in base._field_defaults:
                        namespace.setdefault(name, base._field_defaults[name])
        fields.update(cur_fields)
        namespace['__annotations__'] = fields

        # Let `NamedTupleMeta` create a annotated `namedtuple` for us.
        # Note that `bases` is not used there so we just set it to `None`.
        nm_tpl = super().__new__(mcs, typename, None, namespace)
        return nm_tpl

class Options(metaclass=OptionsMeta):
    _root = True

    def __new__(cls, *args, **kwargs):
        if cls is Options:
            # Prevent instantiation of `Options` class.
            raise TypeError("Type Options cannot be instantiated; "
                            "it can be used only as a base class")
        return super().__new__(cls, *args, **kwargs)
```

A few things to notice here:

- We define a new metaclass that inherits `NamedTupleMeta` so we could call its `__new__` method that takes care of everything for us. The `Options` class doesn't really do anything, and for simplicity, we forbid directly instantiating it like we could for `NamedTuple`.
- `annotations` must be an `OrderedDict` because the ordering of fields matter — the order determines the index of the field in the underlying tuple object. Here we put base class fields in front of derived ones, but leave out ones that are redefined.
- A limitation of this method is that the base class cannot contain fields with default values, unless: *a)* they're redefined in the base class, or *b)* every field in the derived class also comes with a default value.

If you understood what we've learnt so far, the implementation is actually pretty straightforward. However, we encounter problems when we try to use it in practice:

```python
In [1]: class BaseOptions(Options):
   ...:     a: int
   ...:     b: int = 2

In [2]: class DerivedOptions(BaseOptions):
   ...:     b: float = 0.5
   ...:     c: float = 1.0

In [3]: BaseOptions(1)
Out[3]: BaseOptions(a=1, b=2)

In [4]: DerivedOptions(2)
Out[4]: BaseOptions(a=2, b=2)

In [5]: DerivedOptions(2, 0.3, 0.4)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-5-f4db6b51352e> in <module>
----> 1 DerivedOptions(2, 0.3, 0.4)

TypeError: __new__() takes from 2 to 3 positional arguments but 4 were given
```

The error message may seem a bit cryptic, but what happens here is that `DerivedOptions` became an alias for `BaseOptions`. A deeper investigation shows that `OptionsMeta.__new__` is not even called when `DerivedOptions` is created. How come?

The truth is, the `nm_tpl` returned from the constructor of `NamedTuple` is of type `collections.namedtuple`, and of course, the metaclass of which is not `OptionsMeta`. When inheriting the `nm_tpl` class, we're actually inheriting a namedtuple, not an `Options` subclass.

Thus, we must create a new class using the namespace of `nm_tpl`, and we do so by directly invoking the `__new__` method of `type`, which is `NamedTupleMeta`'s super class:

```python
return type.__new__(mcs, typename, bases + nm_tpl.__bases__, nm_tpl.__dict__.copy())
```

To explain this method call:

- `type.__new__` will create a class with metaclass set to `mcs` (which is `OptionsMeta` in this case).

- An added benefit here is that we get to set the base class of the created class, in this case, `BaseOptions` (from `bases`) and `tuple` (from `nm_tpl.__bases__`). Note that it's essential to keep `tuple` a base class, because `tuple.__new__` is called when we create an instance of this namedtuple, and that requires the class to be a subclass of `tuple`. If we don't do that, we get an exception:

  ```python
  TypeError: tuple.__new__(DerivedOptions): DerivedOptions is not a subtype of tuple
  ```

- The `__dict__` (namespace) of `nm_tpl` is used as is. We do a copy because `type.__new__` requires this namespace dictionary to be writable (of type `dict`), but `__dict__` is not (of type `mappingproxy`).

Since we were able to keep the actual base class (`BaseOptions`) in the MRO[^3] of the derived class, Python automatically takes care of the latter two functionalities we wanted to accomplish by inheritance. We can easily verify this:

```python
In [1]: class BaseOptions(Options):
   ...:     a: int
   ...:     @property
   ...:     def foo(self):
   ...:         return self.a

In [2]: class DerivedOptions(BaseOptions):
   ...:     b :int

In [3]: x = DerivedOptions(1, 2)

In [4]: x.foo
Out[4]: 1

In [5]: isinstance(x, BaseOptions)
Out[5]: True
```

[^3]: The MRO (method resolution order) is Python's answer to the diamond dependency problem in multiple inheritance. When we access a method of an instance, we find the first class in its MRO that defines such method, and returns the method of that class. In the single inheritance case, MRO can be thought of as the list of ancestor classes from the derived class to `object`, the base class of everything. Please refer to [this Wikipedia article](https://en.wikipedia.org/wiki/C3_linearization) for the algorithm used to compute MRO — the C3 linearization algorithm.

## Multiple Inheritance

The method above also fits for multiple inheritance — we just need to gather fields from all the base classes. However, with multiple bases come other problems that did not exist in the single inheritance case:

- What if multiple base classes define the same field? Since we're exploring uncharted waters here, we get to define the behavior, but it has to be intuitive. My opinion is that base classes must not have overlapping fields, unless they're redefined in the derived class. This guarantees that there aren't unexpected overwrites of fields by different orderings of the base classes. But of course, if you implement it, you're free to choose whatever strategy that pleases you.
- What if a base class is not a subclass of `Options`? We should still keep it `bases` so it's kept in the MRO[^4], and instances could access its methods.

[^4]: If you don't know what this means, you have skipped [footnote 3](#fn:3).

Now, let's try implementing this `OptionsMeta` metaclass that supports multiple inheritance:

```python
class OptionsMeta(typing.NamedTupleMeta):
    def __new__(mcs, typename, bases, namespace):
        if namespace.get('_root', False):
            # The created class is `Options`, skip.
            return super().__new__(mcs, typename, bases, namespace)

        # Gather fields from annotations of current class and base classes.
        cur_fields = namespace.get('__annotations__', {})
        fields = collections.OrderedDict()
        field_sources = {}  # which base class does the name came from
        field_defaults = {}
        for base in bases:
            if issubclass(base, Options) and hasattr(base, '_fields'):
                # Base class is a concrete subclass of `Options`.
                for name in base._fields:
                    if name in cur_fields:
                        # Make sure not to overwrite redefined fields.
                        continue
                    if name in fields:
                        # Overlapping field that is not redefined.
                        raise TypeError(
                            f"Base class {base} contains field {name}, which "
                            f"is defined in other base class "
                            f"{field_sources[name]}")
                    fields[name] = base.__annotations__[name]
                    field_sources[name] = base
                    if name in base._field_defaults:
                        field_defaults[name] = base._field_defaults[name]
        fields.update(cur_fields)
        if len(fields) == 0:
            raise ValueError("Options class must contain at least one field")
        for name, value in field_defaults.items():
            namespace.setdefault(name, value)
        namespace['__annotations__'] = fields

        # Let `NamedTupleMeta` create a annotated `namedtuple` for us.
        # Note that `bases` is not used here so we just set it to `None`.
        print(fields)
        nm_tpl = super().__new__(mcs, typename, None, namespace)

        # Wrap the return type in `OptionsMeta` so it can be subclassed.
        # Also keep base classes of the `namedtuple` (i.e., the `tuple` class),
        # so we can call `tuple.__new__`.
        bases = bases + nm_tpl.__bases__
        return type.__new__(mcs, typename, bases, nm_tpl.__dict__.copy())
```

This works great when we inherit from non-`Options` classes, as we can see from these examples:

```python
In [1]: class BaseOptions(Options):
   ...:     a: int
   ...:     @property
   ...:     def foo(self):
   ...:         return self.a

In [2]: class Mixin:
   ...:     def bar(self):
   ...:         return self.a + self.b

In [3]: class DerivedOptions(BaseOptions, Mixin):
   ...:     b :int

In [4]: x = DerivedOptions(1, 2)

In [5]: x.foo
Out[5]: 1

In [6]: x.bar()
Out[6]: 3

In [7]: isinstance(x, BaseOptions)
Out[7]: True

In [8]: isinstance(x, Mixin)
Out[8]: True
```

But when we try to inherit from two `Options` subclasses, something weird happens:

```python
In [1]: class OptionsA(Options):
   ...:     a: int
   ...:     b: int

In [2]: class OptionsB(Options):
   ...:     c: int
   ...:     d: int

In [3]: class MergedOptions(OptionsA, OptionsB):
   ...:     pass
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-3-51d384fffb01> in <module>
----> 1 class MergedOptions(OptionsA, OptionsB):
      2     pass
      3

<ipython-input-3-5ff213f4a3b5> in __new__(mcs, typename, bases, namespace)
     43         # so we can call `tuple.__new__`.
     44         bases = bases + nm_tpl.__bases__
---> 45         return type.__new__(mcs, typename, bases, nm_tpl.__dict__.copy())
     46

TypeError: multiple bases have instance lay-out conflict
```

Now this is something new, an error message I've never seen before. It turns out that I cannot inherit from multiple built-in classes that don't go together at the C level[^5], in this case, two different subclasses of `tuple`. I can see why this is a problem: such built-in types are implemented in C, with fixed memory layouts and implementations for special methods.

[^5]: This is a simplified explanation. [This StackOverflow answer](https://stackoverflow.com/questions/48136025/typeerror-multiple-bases-have-instance-lay-out-conflict) gave a pointer to the CPython source code that calculates the best "solid base" for a new class. I'm not familiar with CPython implementations, but my guess is that the solid base is the first class among the MRO with a memory layout different from its base class. Note that adding Python attributes and methods don't affect the memory layout, because that's equivalent to adding entries to the `__dict__` dictionary.

    Also note that this is not limited to CPython. Mypy also has [a similar check](https://bitbucket.org/pypy/pypy/annotate/default/pypy/objspace/std/typeobject.py?at=default&fileviewer=file-view-default#typeobject.py-1064:1086).

If we can't create the type with our bases, how about modifying the bases after creation? It turns out you can't do that either:

```python
<ipython-input-118-d6cd3ab74257> in __new__(mcs, typename, bases, namespace)
     43         # so we can call `tuple.__new__`.
     44         options_type = type.__new__(mcs, typename, nm_tpl.__bases__, nm_tpl.__dict__.copy())
---> 45         options_type.__bases__ = bases
     46         return options_type
     47

TypeError: __bases__ assignment: 'Options' object layout differs from 'tuple'
```

It seems that we're out of luck. But actually, here's some less known evil: you can [override the creation of the MRO](http://stupidpythonideas.blogspot.com/2015/12/can-you-customize-method-resolution.html) in the metaclass! But the crazy thing here is, we need to implement the C3 linearization algorithm ourselves. Luckily, it's a simple algorithm:

```python
class OptionsMeta(typing.NamedTupleMeta):
    def __new__(mcs, typename, bases, namespace):
        ...  # omitted here
        new_namespace = nm_tpl.__dict__.copy()
        new_namespace['_bases'] = bases
        options_type = type.__new__(mcs, typename, nm_tpl.__bases__, new_namespace)
        # Writing to `__bases__` triggers an MRO update. This has to be done after
        # class creation because otherwise we can't access `_bases`.
        options_type.__bases__ = tuple(nm_tpl.__bases__)
        return options_type

    def mro(cls):
        default_mro = super().mro()
        # `Options` does not define `_bases`, so we don't do anything about it.
        if hasattr(cls, '_bases'):
            # `default_mro` should be `[cls, tuple, object]`.
            # `c3merge` and `c3mro` are implementations of the C3 linearization
            # algorithm, which unluckily aren't provided as APIs.
            return c3merge([
                default_mro[:1],
                *[base.__mro__ for base in cls._bases],
                default_mro[1:]])
        return default_mro

def c3merge(sequences):
    r"""Adapted from https://www.python.org/download/releases/2.3/mro/"""
    # Make sure we don't actually mutate anything we are getting as input.
    sequences = [list(x) for x in sequences]
    result = []
    while True:
        # Clear out blank sequences.
        sequences = [x for x in sequences if x]
        if not sequences:
            return result
        # Find the first clean head.
        for seq in sequences:
            head = seq[0]
            # If this is not a bad head (i.e., not in any other sequence)
            if not any(head in s[1:] for s in sequences):
                break
        else:
            raise Error("inconsistent hierarchy")
        # Move the head from the front of all sequences to the end of results.
        result.append(head)
        for seq in sequences:
            if seq[0] == head:
                del seq[0]
    return result
```

Of course, this complex method is when you need to support every general case. Normally you wouldn't have multiple layers of hierarchy for namedtuples, nor will you mix-in a bunch of other classes such that you need to be careful about the MRO.

## Arbitrary Order of Fields

Now, to the final goal which you've probably forgotten: removing the constraint on ordering for fields with default values. This is an inherent limit in Python, because method arguments with default values are treated as keyword arguments (captured by `**kwargs`), and have to be declared after positional arguments (captured by `*args`).

To workaround this, we can declare all arguments of the constructor as keyword-only arguments. For me, not allowing positional arguments is actually better because the order of the fields can be ambiguous when you have multiple base classes.

How can we programmatically create a method with custom arguments? Let's dive into the code for `collections.namedtuple`, where the magic happens. The code is pretty long so I'm just going to show the relevant parts here. Turns out magic doesn't exist, everything's just a hack:

```python
    ...  # omitted
    arg_list = repr(field_names).replace("'", "")[1:-1]

    # Create all the named tuple methods to be added to the class namespace

    s = f'def __new__(_cls, {arg_list}): return _tuple_new(_cls, ({arg_list}))'
    namespace = {'_tuple_new': tuple_new, '__name__': f'namedtuple_{typename}'}
    # Note: exec() has the side-effect of interning the field names
    exec(s, namespace)
    __new__ = namespace['__new__']
    __new__.__doc__ = f'Create new instance of {typename}({arg_list})'
    if defaults is not None:
        __new__.__defaults__ = defaults
    __new__.__qualname__ = f'{typename}.__new__'

    ...  # omitted
    class_namespace = {
        ...  # omitted
        '__new__': __new__,
    }

    ...  # omitted
    result = type(typename, (tuple,), class_namespace)

    ...  # omitted
```

Yep, that's right. The `__new__` method for the namedtuple is created by *writing code as a string and calling `exec`*. To be honest, that's probably the easiest way, and we shouldn't have gone this far if we need to talk about elegant and readable implementations.

Following their lead, we can also create our own version of `__new__` and overwrite theirs:

```python
        # Rewrite `__new__` method to make all arguments keyword-only.
        # This is very hacky code. Do not try this at home.
        arg_list = ''.join(name + ', '  # watch out for singleton tuples
                           for name in reordered_fields)
        s = (f"""
        def __new__(_cls, *args, {arg_list}):
            if len(args) > 0:
                raise TypeError("Instances of Options class must be created "
                                "with keyword arguments.")
            return _tuple_new(_cls, ({arg_list}))
        """).strip()  # remove incorrect indents in the string
        new_method_namespace = {'_tuple_new': tuple.__new__,
                                '__name__': f'namedtuple_{typename}'}
        exec(s, new_method_namespace)
        __new__ = new_method_namespace['__new__']
        __new__.__qualname__ = f'{typename}.__new__'
        __new__.__doc__ = nm_tpl.__new__.__doc__
        __new__.__annotations__ = nm_tpl.__new__.__annotations__
        __new__.__kwdefaults__ = {name: namespace[name]
                                  for name in fields_with_default}
        nm_tpl.__new__ = __new__
```

As the comment says, this is very dangerous. Don't try this at home.

## Summary

So far, we've delivered our promises. We have a super-enhanced version of namedtuple that supports multiple inheritance and arbitrary field orders. You can find the entire working code in [this GitHub Gist](https://gist.github.com/huzecong/df51502a8a6ec0bcc0e605a2ce109008). It's a bit long, but you don't really need to know the details — do the Pythonic thing and treat it as library.

But you may ask, what's it useful for?

I dunno, but it's a pretty fun journey, isn't it?
