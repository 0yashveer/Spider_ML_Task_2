# Role of each gate
  A basic LSTM contains two memory blocks , one is Long term memory stored in the 'Cell State' and another 
  is short term memory 'hiddem state'.
    
  ## Forget Gate 
  Forget Gate is responsible for managing the Long term context of a sentence. As its name suggests , it tells the system by telling which information to forget by
  by giving a simple probability (0-1) bu using sigmodial activation for different dims , its dim are that of the Cell State.

  ## Input Gate 
  Input Gate determines what part of the new input would be kept in the Cell State for the Long term context.

  ## Output Gate
  Output gate is responsible for selecting what part of the cell state would be beneficial for forming the current hidden state .     
