
let width = 400;
let height = 400;

let cell_width = 50;
let rows, cols;

let nnmodel;
let gen = 0;

function setup() {
    createCanvas(width, height);

    rows = width / cell_width;
    cols = height / cell_width;

    nnmodel = new NNModel();
    console.log( "model compiled" );

}

function draw(){
    train_model( 100 )
         .then( () => drawXOR() );

    noLoop();

}

async function drawXOR() {

    gen++;

    document.querySelector('#gen').innerHTML = 'Generation #' + gen;

    for( var rid = 0; rid < rows; rid++ ){
        for( var cid = 0; cid < cols; cid++ ){

            var ip_x = rid / rows ;
            var ip_y = cid / cols ;

            var resp = await nnmodel.predict( ip_x, ip_y, );
            draw_pixel( rid, cid, resp * 255 );
        }
    }

    loop();
}

async function train_model( iters ){
    for( var id = 0; id < iters; id++ ){
        await nnmodel.train();
    }
}

function predict( rid, cid ){
    return random( 100, 255 );
}

function draw_pixel( rid, cid, pred_color ){
    push();
    noStroke();
    fill( pred_color );
    rect( rid * cell_width, cid * cell_width, cell_width, cell_width );
    fill( 255 );
    text( Math.ceil(pred_color), ( rid * cell_width ) + cell_width / 3, ( cid * cell_width ) + cell_width / 2 )
    pop();
}