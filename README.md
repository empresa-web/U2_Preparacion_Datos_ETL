# Preparación de datos – Unidad 2

Caso de estudio: ventas simuladas de una tienda comercial. Todos los registros son ficticios y solo se utilizan con fines académicos.

## Archivos
- `datos/ventas_original.csv`: 122 filas de origen, con errores intencionales.
- `datos/ventas_limpias.csv`: 115 filas luego de ETL.
- `scripts/limpieza_datos.py`: script reproducible con Python 3, sin dependencias externas.
- `evidencias/reporte_limpieza.json`: estadísticas verificables del proceso.
- `sql/data_warehouse.sql`: definición de esquema estrella MySQL.
- `sql/carga_datos.sql`: instrucciones para cargar tablas dimensionales y hechos.

## Cómo reproducir
1. Desde la carpeta principal, ejecutar `python scripts/limpieza_datos.py`.
2. Verificar que se regeneren `ventas_limpias.csv` y `reporte_limpieza.json`.
3. Ejecutar primero `sql/data_warehouse.sql` en MySQL 8.0+.
4. Ejecutar después `sql/carga_datos.sql` en la misma instancia MySQL.
5. Revisar la consulta de verificación incluida al final del segundo script SQL.

## Resultados del ejemplo
- Filas originales: 122
- Filas conservadas: 115
- Filas descartadas: 7
- Suma de ventas en datos limpios: $910,320.00 MXN

**Alcance:** ejecución del ETL en Python verificada; la carga SQL se entrega preparada pero no se ha ejecutado en el servidor MySQL del estudiante.
