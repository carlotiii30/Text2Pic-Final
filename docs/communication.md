# Comunicación entre Java y Python
Para la conexión entre Java y Python existen distintas opciones:
1. Jython.
2. Sockets.
3. Runtime.
4. Servicios web.

## Opciones
### Jython
Es una implementación de Python que se ejecuta sobre la JVM. Se puede ejecutar
código Python directamente desde Java.

Es ideal si se necesita integrar código Python en una aplicación Java existente.
Sin embargo, solo soporta hasta Python 2.7.

### Sockets
Proporcionan una forma estándar de comunicación entre procesos en diferentes
sistemas. Se puede crear un servidor en Python que escuche en un socker y enviar
datos desde Java a través de la red.

Esta opción es útil cuando se necesita comunicación entre programas en máquinas
diferentes o cuando se busca una arquitectura cliente-servidor.

### Runtime
La clase `Runtime` en Java permite ejecutar comandos en el sistema operativo
subyacente. Se puede invocar un script de Python desde Java utilizando líneas
de comandos.

### Servicios web
Se puede crear un servicio web utilizando un framework como Flask o Django en
Python y consumirlo desde Java utilizando bibliotecas.

Esta opción es más útil cuando se necesita comunicación entre aplicaciones
distribuidas y una arquitectura más flexible y escalable.

## Decisión final
Para este proyecto utilizaremos Sockets: Estableceremos un protocolo de
comunicación y crearemos un servidor en Python y un cliente en Java.