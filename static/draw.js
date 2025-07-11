const cvs = document.getElementById('board');
const ctx = cvs.getContext('2d');
ctx.fillStyle='black';ctx.fillRect(0,0,280,280);
ctx.lineCap='round';ctx.lineWidth=20;ctx.strokeStyle='white';

let drawing=false;
cvs.onmousedown=()=>{drawing=true;ctx.beginPath();};
cvs.onmouseup=()=>drawing=false;
cvs.onmousemove=e=>{
  if(!drawing) return;
  const r=cvs.getBoundingClientRect();
  ctx.lineTo(e.clientX-r.left,e.clientY-r.top);
  ctx.stroke();
};

document.getElementById('clear').onclick=()=>{
  ctx.fillStyle='black';ctx.fillRect(0,0,280,280);
};

document.getElementById('predict').onclick=async()=>{
  const dataURL=cvs.toDataURL('image/png');
  const res=await fetch('/predict',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:dataURL})
  });
  const {digit}=await res.json();
  document.getElementById('result').innerText=`模型预测: ${digit}`;
};
