<script lang="ts">
    const n = 25
    const r = [5,5,10,5]
    const c = [10,5,5,5]
    const a = [
        [[0,1],[0]],
        [[2],[1,2]],
        [[3],[3]]
    ]

    const w = 25
    const h = 25

    const ry = 150

    function* segs(sgmt: number[]){
        let s = 0
        for (const size of sgmt){
            const start = s
            const stop = s + size
            s = stop

            yield [start,stop]
        }
    }

    function* elems(start:number,stop:number){
        for(let i = start; i<stop; i+=1){
            yield i
        }
    }

    function* pairs(a: [number[],number[]][]){
        
    }

    function* lines(r:number,c:number, a: [number[],number[]][]){
        for(let i = start; i<stop; i+=1){
            yield i
        }
    }

</script>

<h1>Segmentation</h1>
<svg viewBox="-0.5 -0.5 1000 1000">
    <!-- reference segmentation (r) -->
    <rect width={w*n} height={h} stroke="black" fill-opacity="0"/>
    {#each segs(r) as [start,stop]}
        {#each elems(start,stop) as e}
            {#if e>0 && e===start}
                <line x1={e*w} x2={e*w} y1={h*0} y2={h*0+h} stroke="black" stroke-width="2"/>
            {:else if e>0 && e!==start}
                <line x1={e*w} x2={e*w} y1={h*0} y2={h*0+h} stroke="black" stroke-dasharray="1,1"/>
            {/if}
        {/each}
    {/each}

    <!-- alignment (a) -->

    <!-- candidate segmentation (c) -->
    <rect width={w*n} height={h} y={h*3} stroke="black" fill-opacity="0"/>
    {#each segs(c) as [start,stop]}
        {#each elems(start,stop) as e}
            {#if e>0 && e===start}
                <line x1={e*w} x2={e*w} y1={h*3} y2={h*3+h} stroke="black" stroke-width="2"/>
            {:else if e>0 && e!==start}
                <line x1={e*w} x2={e*w} y1={h*3} y2={h*3+h} stroke="black" stroke-dasharray="1,1"/>
            {/if}
        {/each}
    {/each}
</svg>
