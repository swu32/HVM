# HSTC

now we have a function that checks whether the chunk has occurred in the
sequence, we could iterate through the sequence and use the pre-existing chunks to find whether there is a corresponding match.
     
reveal the new observation,check the observation up to time T
matches with the explanation of chunks. =============== Perception of Chunks ===============
the definition of the currrent chunk is all chunks which ends at this time point t.


chunk termination time: time when the last chunk in this dimension terminates,
explanation in that dimension stopped, any observation afterwards needs to be explained by other means.
span through the sequence to find if there is a template that matches with the chunk item
element is at t-1
previous chunks are the chunks that ended between the initiation of
the current chunk, and the end of this current chunk, which is the time point at the moment.

Every moment as soon as a chunk ends, the transition between this chunk and the other chunks
 are then to use to suggest an agglomerative chunk.

have a record of which chunks are ending between time t and time t - L,
L is the longest temporal length of each chunk

update transitions between chunks which as a temporal proximity
Update only happens when a chunk has officially identified to end at the moment t
search in previous_chunk_boundary_record to find chunks that are temporally close to the current chunk
is the starting location of this chunk.
takes record of which chunk has been associated with the current chunk
this point include all the chunks that ends right before t - temporal_length_of_current_chunk


every time when a chunk is identified to finish, it updates transition and marginal probability and run a round of 
chunk decision 