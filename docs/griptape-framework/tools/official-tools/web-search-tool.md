# Web Search Tool

This tool enables LLMs to search the web.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/web_search_tool_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/web_search_tool_1.txt"
    ```


```
[09/08/23 15:37:25] INFO     Task 2cf557f7f7cd4a20a7fa2f0c46af2f71              
                             Input: Tell me how photosynthesis works            
[09/08/23 15:37:32] INFO     Subtask d023ef9f41d142229513510cf4f47afe           
                             Thought: I know that photosynthesis is a process   
                             used by plants and other organisms to convert light
                             energy into chemical energy that can later be      
                             released to fuel the organisms' activities.        
                             However, to provide a detailed explanation, I will 
                             need to conduct a web search.                      
                                                                                
                             Action: {"name": "WebSearch",      
                             "path": "search", "input": {"values": {"query":
                             "How does photosynthesis work?"}}}                 
                    INFO     Subtask d023ef9f41d142229513510cf4f47afe           
                             Response: {'url':                               
                             'https://www.nationalgeographic.org/encyclopedia/ph
                             otosynthesis/', 'title': 'Photosynthesis',         
                             'description': 'Jul 15, 2022 ... Photosynthesis is 
                             the process by which plants use sunlight, water,   
                             and carbon dioxide to create oxygen and energy in  
                             the form of sugar.'}                               
                             {'url':                                            
                             'https://www.snexplores.org/article/explainer-how-p
                             hotosynthesis-works', 'title': 'Explainer: How     
                             photosynthesis works', 'description': 'Oct 28, 2020
                             ... Photosynthesis is the process of creating sugar
                             and oxygen from carbon dioxide, water and sunlight.
                             It happens through a long series of                
                             chemical\xa0...'}                                  
                             {'url':                                            
                             'https://www.sciencefocus.com/nature/how-does-photo
                             synthesis-work', 'title': 'Photosynthesis: What is 
                             it and how does it work? - BBC Science ...',       
                             'description': "Jul 27, 2022 ... Photosynthesis is 
                             the process by which carbohydrate molecules are    
                             synthesised. It's used by plants, algae and certain
                             bacteriato turn sunlight,\xa0..."}                 
                             {'url': 'https://oregonforests.org/photosynthesis',
                             'title': 'Photosynthesis | OregonForests',         
                             'description': "Here's how it works: Tree and plant
                             roots absorb water, as well as minerals and        
                             nutrients, from the soil. At the same time, the    
                             leaves or needles absorb carbon\xa0..."}           
                             {'url':                                            
                             'https://ssec.si.edu/stemvisions-blog/what-photosyn
                             thesis', 'title': 'What is Photosynthesis |        
                             Smithsonian Science Education Center',             
                             'description': 'Apr 12, 2017 ... Rather, plants use
                             sunlight, water, and the gases in the air to make  
                             glucose, which is a form of sugar that plants need 
                             to survive. This process\xa0...'}                  
[09/08/23 15:37:50] INFO     Task 2cf557f7f7cd4a20a7fa2f0c46af2f71              
                             Output: Photosynthesis is the process by which     
                             plants, algae, and certain bacteria convert light  
                             energy, usually from the sun, into chemical energy 
                             in the form of glucose or sugar. This process      
                             involves several steps:                            
                                                                                
                             1. Absorption of light: The process begins when    
                             light is absorbed by proteins containing           
                             chlorophylls (pigments) present in chloroplasts.   
                                                                                
                             2. Conversion of light energy to chemical energy:  
                             The absorbed light energy is used to convert carbon
                             dioxide from the atmosphere and water from the soil
                             into glucose. This conversion process occurs       
                             through a series of chemical reactions known as the
                             light-dependent reactions and the Calvin cycle.    
                                                                                
                             3. Release of oxygen: As a byproduct of these      
                             reactions, oxygen is produced and released into the
                             atmosphere.                                        
                                                                                
                             4. Use of glucose: The glucose produced is used by 
                             the plant for growth and development. It can also  
                             be stored for later use.                           
                                                                                
                             In summary, photosynthesis is a vital process for  
                             life on Earth as it is the primary source of oxygen
                             in the atmosphere and forms the basis of the food  
                             chain.      
```

Extra schema properties can be added to the Tool to allow for more customization if the Driver supports it.
In this example, we add a `sort` property to the `search` Activity which will be added as a [Google custom search query parameter](https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/tools/official-tools/src/web_search_tool_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/tools/official-tools/logs/web_search_tool_2.txt"
    ```

