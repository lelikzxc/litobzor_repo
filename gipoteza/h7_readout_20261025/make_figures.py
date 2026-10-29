"""Static research figures from completed frozen outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def main():
    out=BASE/'figures';out.mkdir(exist_ok=True)
    j=json.loads((BASE/'cost_extension/SUMMARY.json').read_text());geos=sorted(j['methods']['center']['by_geometry'])
    center=np.array([j['methods']['center']['by_geometry'][g] for g in geos])
    path=np.array([j['methods']['witness']['by_geometry'][g] for g in geos])
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');x=np.arange(len(geos));width=.36
    ax.bar(x-width/2,center,width,label='Центральная эвристика',color='#6b7280')
    ax.bar(x+width/2,path,width,label='Геометрическое свидетельство',color='#2563eb')
    ax.set_xticks(x,[g[:6] for g in geos],rotation=35,ha='right');ax.set_xlabel('Геометрия: первые 6 символов хеша')
    ax.set_ylabel('Уточняющих блоков на локальное событие')
    ax.set_title('10 других геометрий: точное совпадение всех классов\nЭкономия уточнений 35,43%; групповой 95% интервал [31,23; 39,19]%')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'cost_transfer.png',dpi=180);fig.savefig(out/'cost_transfer.svg');plt.close(fig)
    p=np.load(BASE/'mechanism_examples.npz')
    selected=[];seen=set()
    for i,case in enumerate(p['case_index']):
        if int(case) not in seen:selected.append(i);seen.add(int(case))
        if len(selected)==3:break
    n=len(selected);fig,axes=plt.subplots(n,3,figsize=(9,3*n+.5),layout='constrained')
    axes=np.atleast_2d(axes)
    for row,i in enumerate(selected):
        original=p['original'][i];alternative=p['alternative'][i]
        counts=original.reshape(32,8,32,8).sum((1,3))/64
        for ax,im,title in zip(axes[row],[original,counts,alternative],['Исходные детали','Одинаковые агрегаты','Другие допустимые детали']):
            ax.imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            if row==0:ax.set_title(title,fontsize=10)
            ax.axis('off')
        axes[row,0].text(0,-.035,f'Событие {int(p["case_index"][i])}',transform=axes[row,0].transAxes,fontsize=9,va='top')
    fig.suptitle('Неоднозначность частичного бинарного наблюдения\nПерестановки внутри блоков — не физическая симуляция литографии',fontsize=11)
    fig.savefig(out/'ambiguity.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
