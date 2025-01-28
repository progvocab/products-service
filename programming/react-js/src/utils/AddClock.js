
import React, {useState , useEffect } from "react";

const AddClock =({ show =true , data}) =>{
   
    return (
      
         <button  onClick={()=>data="showClock"} >Add Clock</button>
    );
};

export default AddClock;