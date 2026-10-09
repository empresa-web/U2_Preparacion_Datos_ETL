-- MySQL 8.0+; modelo dimensional para ventas simuladas
CREATE DATABASE IF NOT EXISTS dw_ventas_comerciales CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE dw_ventas_comerciales;
CREATE TABLE IF NOT EXISTS Dim_Producto (
 id_producto INT PRIMARY KEY, nombre_producto VARCHAR(100) NOT NULL,
 categoria VARCHAR(50) NOT NULL, marca VARCHAR(50) NOT NULL,
 descripcion VARCHAR(255), precio_referencia DECIMAL(12,2)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS Dim_Tiempo (
 id_tiempo INT PRIMARY KEY, fecha DATE NOT NULL UNIQUE,
 dia INT NOT NULL, mes INT NOT NULL, trimestre INT NOT NULL, anio INT NOT NULL,
 nombre_mes VARCHAR(20), nombre_dia VARCHAR(20)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS Dim_Sucursal (
 id_sucursal INT PRIMARY KEY, nombre_sucursal VARCHAR(100) NOT NULL,
 ciudad VARCHAR(50) NOT NULL, estado VARCHAR(50) NOT NULL,
 direccion VARCHAR(150), tipo_sucursal VARCHAR(50)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS Dim_Pago (
 id_pago INT PRIMARY KEY, metodo_pago VARCHAR(50) NOT NULL,
 descripcion VARCHAR(100), tipo VARCHAR(50)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS Fact_Ventas (
 id_venta INT PRIMARY KEY, id_producto INT NOT NULL, id_tiempo INT NOT NULL,
 id_sucursal INT NOT NULL, id_pago INT NOT NULL,
 cantidad INT NOT NULL, precio_unitario DECIMAL(12,2) NOT NULL,
 total_venta DECIMAL(12,2) NOT NULL,
 CONSTRAINT fk_ventas_producto FOREIGN KEY(id_producto) REFERENCES Dim_Producto(id_producto),
 CONSTRAINT fk_ventas_tiempo FOREIGN KEY(id_tiempo) REFERENCES Dim_Tiempo(id_tiempo),
 CONSTRAINT fk_ventas_sucursal FOREIGN KEY(id_sucursal) REFERENCES Dim_Sucursal(id_sucursal),
 CONSTRAINT fk_ventas_pago FOREIGN KEY(id_pago) REFERENCES Dim_Pago(id_pago),
 INDEX idx_ventas_tiempo(id_tiempo), INDEX idx_ventas_sucursal(id_sucursal)
) ENGINE=InnoDB;
