{% if objtype == 'property' %}
:orphan:
{% endif %}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == 'property' %}
property
{% endif %}

.. auto{{ objtype }}:: {{ fullname | replace("bsparse.", "bsparse::") }}

{# In the fullname (e.g. `bsparse.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `bsparse::methodname`)
specifies `bsparse` as the module name. #}