# Gestor de dependencias

La administración de dependencias es el proceso de automatización de la instalación, actualización, configuración y eliminación de bibliotecas, paquetes y herramientas de las que depende una aplicación. Cada lenguaje de programación tiene sus propias herramientas de gestión de dependencias, por tanto, deberemos tener en cuenta que el lenguaje de programación que vamos a utilizar en este proyecto es Python.

Las opciones que vamos a explorar son:
* Poetry: https://python-poetry.org/
* pip + virtualenv: https://pip.pypa.io/ y https://virtualenv.pypa.io/
* Conda: https://docs.conda.io/


## Criterios de elección
Vamos a establecer unos criterios para poder elegir la mejor opción para nuestro proyecto.
- **Seguridad**: Capacidad de garantizar que las bibliotecas y paquetes utilizados no contengan vulnerabilidades conocidas.
- **Estabilidad**: Capacidad de garantizar que las versiones de las bibliotecas y paquetes utilizados en un proyecto se mantengan constantes.
- **Comunidad**: Cantidad de recursos y paquetes desarrollados, y ayuda en la resolución de problemas comunes.


## Poetry
- **Todo en uno**: Combina la gestión de dependencias, la creación de entornos virtuales y la publicación de paquetes en una sola herramienta.
- **Archivo de bloqueo**: Utiliza un archivo `poetry.lock` para garantizar que las dependencias sean consistentes en todos los entornos.
- **Administración de versiones**: Simplifica el manejo de versiones de las dependencias, asegurando compatibilidad y estabilidad.
- **Publicación de paquetes**: Facilita la creación y publicación de paquetes en PyPI.
- **Interfaz amigable**: Tiene una interfaz de línea de comandos intuitiva y fácil de usar.
- **Resolución de dependencias**: Resuelve automáticamente las dependencias y sus versiones.

## pip + virtualenv
- **Popularidad**: pip es el gestor de paquetes más utilizado en la comunidad Python, lo que asegura una amplia disponibilidad de paquetes.
- **Virtualenv**: Permite la creación de entornos virtuales aislados para gestionar dependencias específicas del proyecto.
- **Simplicidad**: Facilita la instalación, actualización y desinstalación de paquetes mediante comandos simples.
- **Flexibilidad**: Al combinar pip con virtualenv, se obtiene un control granular sobre las dependencias y entornos.
- **Amplia biblioteca de paquetes**: Acceso a una gran cantidad de paquetes en el Python Package Index (PyPI).

## Conda
- **Gestión de entornos y paquetes**: Conda gestiona tanto entornos como dependencias, lo que permite una administración integral.
- **Multilenguaje**: No se limita solo a Python, también puede gestionar paquetes y entornos de otros lenguajes como R.
- **Entornos reproducibles**: Utiliza un archivo `environment.yml` para garantizar que los entornos sean reproducibles.
- **Optimización de dependencias**: Descarga y resuelve eficientemente las dependencias.
- **Canales personalizados**: Permite utilizar canales personalizados para la descarga de paquetes.

## Conclusión
Tanto [Poetry](#poetry), [pip + virtualenv](#pip--virtualenv) y [Conda](#conda) tienen sus puntos fuertes y débiles. Mientras que pip + virtualenv tiene una gran popularidad y flexibilidad, influenciada por la simplicidad y la gran cantidad de paquetes disponibles; Poetry destaca por su enfoque todo en uno y la facilidad de uso, lo que contribuye a la estabilidad y la gestión de versiones. Por otro lado, Conda ofrece una solución integral y multilenguaje, siendo ideal para proyectos que requieran entornos reproducibles y una gestión eficiente de dependencias.

En este proyecto, vamos a utilizar Poetry.