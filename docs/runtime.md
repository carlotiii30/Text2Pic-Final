# Entorno de Ejecución

Un entorno de ejecución es un conjunto de recursos en el cual los programas se ejecutan, proporcionando el soporte necesario para convertir el código fuente en resultados tangibles.

En el contexto de Python, el entorno de ejecución es fundamental para interpretar y ejecutar el código Python.

Las opciones que vamos a explorar son:
* CPython: https://www.python.org
* PyPy: https://www.pypy.org
* Poetry: https://python-poetry.org

## CPython
CPython es la implementación de referencia de Python, desarrollada en C. Es conocida por su robustez y compatibilidad con la mayoría de las bibliotecas y frameworks de Python.

### Características
- **Implementación de Referencia**: Es la implementación estándar de Python.
- **Amplia Compatibilidad**: Compatible con la mayoría de las bibliotecas y frameworks de Python.
- **Estabilidad y Madurez**: Lleva muchos años en desarrollo y es ampliamente utilizado en la industria.
- **Buena Gestión de Memoria**: Utiliza un recolector de basura eficiente para la gestión automática de memoria.

## PyPy
PyPy es una implementación alternativa de Python que se centra en la velocidad y eficiencia. Está escrita en Python y en RPython, un subconjunto de Python.

### Características
- **Alta Velocidad**: Generalmente más rápido que CPython para muchas aplicaciones debido a su compilación JIT (Just-In-Time).
- **Soporte para CPython**: Compatible con la mayoría de las bibliotecas de Python, aunque puede haber excepciones.
- **Menor Uso de Memoria**: Tiene un recolector de basura que puede ser más eficiente en ciertos escenarios.
- **Mejoras en Rendimiento**: Puede mejorar significativamente el rendimiento de aplicaciones Python intensivas en CPU.

## Poetry
Poetry es una herramienta integral para la gestión de dependencias, creación de entornos virtuales y automatización de tareas en proyectos Python.

### Características
- **Gestión de Dependencias**: Permite declarar y gestionar dependencias en un archivo `pyproject.toml`.
- **Creación de Entornos Virtuales**: Facilita la creación de entornos virtuales para aislar dependencias.
- **Automatización de Tareas**: Puede definir y ejecutar tareas personalizadas mediante scripts en el archivo `pyproject.toml`.
- **Archivo de Bloqueo**: Utiliza un archivo `poetry.lock` para garantizar consistencia en las versiones de las dependencias.
- **Publicación de Paquetes**: Facilita la publicación de paquetes en PyPI u otros repositorios.

## Conclusión
Tanto [CPython](#cpython), [PyPy](#pypy) como [Poetry](#poetry) tienen sus ventajas y aplicaciones específicas. CPython es ideal para aplicaciones que requieren una amplia compatibilidad y estabilidad, mientras que PyPy se destaca por su velocidad y eficiencia, especialmente en aplicaciones que se benefician de la compilación JIT. Poetry, por otro lado, es excelente para la gestión de dependencias y automatización de tareas en proyectos Python, proporcionando una experiencia integrada y simplificada.

Para este proyecto, considerando que ya hemos seleccionado Poetry como gestor de dependencias y herramienta de automatización de tareas, también lo elegiremos como opción para el entorno de ejecución Python debido a su integración fluida y capacidades avanzadas de gestión de proyectos.