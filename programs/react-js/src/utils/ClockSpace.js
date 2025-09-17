import React,{useState,useEffect} from 'react'
import AddClock from './AddClock';
const ClockSpace = (data="initial")=>{
    return(
       ( data == "initial")  ? (
            <AddClock></AddClock>    )  :  
        (data == "showClock") ? ( <Clock></Clock> ) : <br/>
    );
};
export default ClockSpace;