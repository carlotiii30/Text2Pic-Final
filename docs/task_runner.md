# Gestor de tareas

Un gestor de tareas es una herramienta de automatización utilizada en el desarrollo de software para ejecutar tareas repetitivas o predefinidas de manera automática. Estas tareas pueden incluir desde la compilación de código hasta la optimización de recursos, la ejecución de pruebas o la organización de activos.

En este caso, debemos considerar que el gestor de dependencias elegido es Poetry.

Las opciones que vamos a explorar son:
* Invoke: https://www.pyinvoke.org/
* Fabric: https://www.fabfile.org/
* Tox: https://tox.readthedocs.io/es/latest/
* Poetry: https://python-poetry.org/

## Criterios de selección
Estableceremos criterios para elegir la mejor opción para nuestro proyecto.
- **Rendimiento**: Velocidad y eficiencia con la que se pueden ejecutar las tareas definidas.
- **Comunidad**: Soporte y respaldo que tiene el gestor de tareas por parte de otros desarrolladores, lo cual puede proporcionar soluciones rápidas a problemas comunes.
- **Flexibilidad**: Capacidad para adaptarse a diferentes necesidades y contextos.

## Invoke
Invoke es una biblioteca para la ejecución de tareas que permite definir y ejecutar tareas desde un archivo `tasks.py`. Es útil para automatizar tareas comunes como la ejecución de pruebas o despliegues.

### Características
- **API sencilla**: Proporciona una API clara y fácil de usar para definir tareas.
- **Flexibilidad**: Permite la definición de flujos de trabajo complejos y dependencias entre tareas.
- **Integración**: Funciona bien con otras herramientas y bibliotecas de Python, facilitando su integración en proyectos existentes.
- **Rendimiento**: Maneja eficientemente la ejecución de tareas y puede ejecutar múltiples tareas en paralelo.

## Fabric
Fabric es una herramienta para simplificar el uso de SSH en tareas de despliegue de aplicaciones o administración de sistemas. Es adecuado para ejecutar comandos en servidores remotos.

### Características
- **Ejecución remota**: Permite la ejecución de comandos en servidores remotos a través de SSH.
- **Scriptabilidad**: Facilita la creación de scripts reutilizables en Python para automatizar tareas de despliegue.
- **Integración**: Se integra fácilmente con otras herramientas y bibliotecas, lo que la convierte en una opción versátil para la automatización.
- **Comunidad**: Cuenta con una comunidad activa y abundante documentación y recursos disponibles.

## Tox
Tox es una herramienta para automatizar pruebas en múltiples entornos. Está diseñada para automatizar y estandarizar las pruebas en diferentes versiones de Python y dependencias.

### Características
- **Gestión de entornos**: Gestiona múltiples entornos de prueba de manera sencilla.
- **Automatización de pruebas**: Automatiza la ejecución de pruebas en diferentes entornos, garantizando compatibilidad y fiabilidad.
- **Integración**: Funciona bien con marcos de prueba populares y tuberías de integración continua (CI/CD).
- **Reproducibilidad**: Asegura resultados de prueba consistentes en diferentes entornos.

## Poetry
Poetry es una herramienta integral para la gestión de dependencias, creación de entornos virtuales y automatización de tareas en proyectos Python.

### Características
- **Gestión de dependencias**: Permite declarar y gestionar dependencias en un archivo `pyproject.toml`.
- **Creación de entornos virtuales**: Facilita la creación de entornos virtuales para aislar dependencias.
- **Automatización de tareas**: Puede definir y ejecutar tareas personalizadas mediante scripts en el archivo `pyproject.toml`.
- **Archivo de bloqueo**: Utiliza un archivo `poetry.lock` para garantizar consistencia en las versiones de las dependencias.
- **Publicación de paquetes**: Facilita la publicación de paquetes en PyPI u otros repositorios.

## Conclusión
Para proyectos Python como este, [Fabric](#fabric) es una excelente opción para ejecución remota y tareas de despliegue, mientras que [Invoke](#invoke) ofrece simplicidad y flexibilidad para definir y ejecutar tareas locales. [Tox](#tox) es ideal para gestionar y automatizar pruebas en múltiples entornos.

Además, [Poetry](#poetry) es una herramienta integral que combina la gestión de dependencias, creación de entornos virtuales y automatización de tareas, lo que la hace una opción sólida para proyectos que requieren simplicidad y eficiencia en el manejo de dependencias y tareas de desarrollo.

Por ser la que hemos elegido como gestor de dependencias, usaremos Poetry.