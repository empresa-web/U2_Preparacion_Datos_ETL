"""ETL reproducible con biblioteca estándar de Python; datos simulados."""
from pathlib import Path
import csv,datetime,json,re,collections
BASE=Path(__file__).resolve().parents[1]
ORIG=BASE/'datos'/'ventas_original.csv'; DEST=BASE/'datos'/'ventas_limpias.csv'; REPORT=BASE/'evidencias'/'reporte_limpieza.json'
PRODUCTOS={1:('Laptop','Electrónica','Lenovo',8500),2:('Mouse','Accesorios','Logitech',250),3:('Teclado','Accesorios','Logitech',470),4:('Monitor','Electrónica','Samsung',3300),5:('Audífonos','Accesorios','Sony',920),6:('Impresora','Electrónica','HP',2600)}
SUCURSALES={1:('Centro','Saltillo','Coahuila'),2:('Norte','Ramos Arizpe','Coahuila'),3:('Sur','Arteaga','Coahuila')}
PAGOS={1:'Efectivo',2:'Tarjeta',3:'Transferencia'}
columns=['id_venta','fecha','id_producto','producto','categoria','marca','id_sucursal','sucursal','ciudad','estado','id_pago','metodo_pago','cantidad','precio_unitario','total_venta']
counts=collections.Counter(); seen=set(); clean=[]
with ORIG.open(encoding='utf-8-sig',newline='') as f:
  raw=list(csv.DictReader(f))
for r in raw:
  try:
    vid=int(r['id_venta']); pid=int(r['id_producto']); sid=int(r['id_sucursal']); mid=int(r['id_pago'])
    if vid in seen: counts['duplicados_eliminados']+=1;continue
    seen.add(vid)
    if pid not in PRODUCTOS or sid not in SUCURSALES or mid not in PAGOS:
      counts['ids_desconocidos_descartados']+=1;continue
    prod,cat,marca,precio_ref=PRODUCTOS[pid]; suc,ciudad,estado=SUCURSALES[sid]; pago=PAGOS[mid]
    ds=r['fecha'].strip()
    try:d=datetime.date.fromisoformat(ds)
    except ValueError:d=datetime.datetime.strptime(ds,'%d/%m/%Y').date();counts['fechas_normalizadas']+=1
    if any([r['producto']!=prod,r['categoria']!=cat,r['marca']!=marca,r['sucursal']!=suc,r['ciudad']!=ciudad,r['estado']!=estado,r['metodo_pago']!=pago]):counts['catalogos_normalizados']+=1
    rawqty=r['cantidad'].strip()
    if not rawqty:counts['cantidades_faltantes_descartadas']+=1;continue
    qty=int(rawqty)
    if qty<=0:counts['cantidades_invalidas_descartadas']+=1;continue
    rawprice=r['precio_unitario'].strip()
    price=float(re.sub(r'[$, ]','',rawprice))
    if price<=0:counts['precios_invalidos_descartados']+=1;continue
    if rawprice!=str(int(price)):counts['precios_normalizados']+=1
    clean.append([vid,d.isoformat(),pid,prod,cat,marca,sid,suc,ciudad,estado,mid,pago,qty,f'{price:.2f}',f'{qty*price:.2f}'])
  except (ValueError,KeyError):counts['registros_con_errores_descartados']+=1
clean.sort(key=lambda r:r[0]);DEST.parent.mkdir(exist_ok=True)
with DEST.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(columns);w.writerows(clean)
report={'origen':'Dataset simulado para fines académicos','registros_originales':len(raw),'registros_limpios':len(clean),'registros_descartados':len(raw)-len(clean),'ventas_total_mxn':round(sum(float(x[-1]) for x in clean),2),'acciones':dict(sorted(counts.items()))}
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
