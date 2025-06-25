import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import './App.css';

function App() {
  const svgRef = useRef();
  const [graphStats, setGraphStats] = useState({ nodes: 0, edges: 0 });
  const [loading, setLoading] = useState(true);
  const [showNodeLabels, setShowNodeLabels] = useState(true);
  const [showRelationLabels, setShowRelationLabels] = useState(false);
  
  useEffect(() => {
    console.log('Starting graph load...');
    setLoading(true);
    
    fetch('/graph_artificial_intelligence.json')
    .then(response => {
      console.log('Fetch response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.text();
    })
    .then(text => {
      console.log('Raw response received');
      try {
        const data = JSON.parse(text);
        if (!data || !data.nodes || !data.links) {
          console.error('Invalid graph data:', data);
          return;
        }
        
        setGraphStats({
          nodes: data.nodes.length,
          edges: data.links.length
        });
        
        console.log(`Graph has ${data.nodes.length} nodes and ${data.links.length} links`);
        
        // Ensure the DOM is ready before rendering
        setTimeout(() => {
          renderGraph(data);
          setLoading(false);
        }, 100);
      } catch (error) {
        console.error('JSON parse error:', error);
        setLoading(false);
      }
    })
    .catch(error => {
      console.error('Failed to load or parse JSON:', error);
      setLoading(false);
    });
  }, []);
  
  // Re-render when toggle states change
  useEffect(() => {
    if (!loading) {
      // Fetch data again and re-render graph with new settings
      fetch('/graph_artificial_intelligence.json')
        .then(response => response.text())
        .then(text => {
          const data = JSON.parse(text);
          renderGraph(data);
        })
        .catch(error => console.error('Failed to reload graph:', error));
    }
  }, [showNodeLabels, showRelationLabels, loading]);
  
  const renderGraph = (data) => {
    if (!svgRef.current) {
      console.error('SVG reference is not available');
      return;
    }

    console.log('Starting graph rendering...');
    const width = window.innerWidth;
    const height = window.innerHeight;
    const nodeRadius = 6;
    // Define safe padding to prevent edge flattening
    const padding = Math.max(width, height) * 0.15; // 15% padding on all sides
    
    // Clear existing SVG content
    const svgElement = d3.select(svgRef.current);
    svgElement.selectAll('*').remove();
    
    console.log('SVG element:', svgElement.node());
    
    // Force simulation with balanced forces
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links)
        .id(d => d.id)
        .distance(70)
        .strength(0.1))
      .force('charge', d3.forceManyBody()
        .strength(-150) // Stronger repulsion
        .distanceMax(500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(nodeRadius * 3).strength(0.8)) // Stronger collision prevention
      .force('x', d3.forceX(width / 2).strength(0.02)) // Weaker center force
      .force('y', d3.forceY(height / 2).strength(0.02)); // Weaker center force
    
    // Add zoom behavior with expanded viewing area
    const zoom = d3.zoom()
      .scaleExtent([0.05, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svgElement.call(zoom);
    
    // Create a group that will contain all the visualization elements
    // Use a larger viewBox to allow nodes to spread beyond the window boundaries
    const g = svgElement.append('g');
    
    // Define arrow markers
    const markerId = `arrowhead-${Date.now()}`;
    const defs = svgElement.append('defs');
    defs.append('marker')
      .attr('id', markerId)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', nodeRadius + 9)
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('xoverflow', 'visible')
      .append('path')
      .attr('d', 'M 0,-3 L 6,0 L 0,3')
      .attr('fill', '#999')
      .attr('stroke', 'none');
    
    // Initialize nodes with a more spread-out position - use a larger radius
    const initialRadius = Math.min(width, height) / 1.5; // Larger initial radius
    data.nodes.forEach((node, i) => {
      const angle = (i / data.nodes.length) * 2 * Math.PI;
      node.x = width/2 + initialRadius * Math.cos(angle);
      node.y = height/2 + initialRadius * Math.sin(angle);
    });
    
    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 1)
      .attr('marker-end', `url(#${markerId})`);
    
    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('r', nodeRadius)
      .attr('fill', '#1f77b4')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);
    
    // Initialize link labels - set visibility based on state
    const linkLabels = g.append('g')
      .attr('class', 'link-labels')
      .style('opacity', showRelationLabels ? 0.7 : 0)
      .selectAll('text')
      .data(data.links)
      .enter()
      .append('text')
      .text(d => d.relation || '')
      .attr('font-size', '8px')
      .attr('fill', '#d3d3d3')
      .attr('text-anchor', 'middle');
    
    // Create node labels with visibility based on state
    const nodeLabels = g.append('g')
      .attr('class', 'node-labels')
      .style('opacity', showNodeLabels ? 1 : 0)
      .selectAll('g')
      .data(data.nodes)
      .enter()
      .append('g')
      .attr('class', 'node-label-group');
    
    // Background rectangle for better label readability
    nodeLabels.append('rect')
      .attr('class', 'label-bg')
      .attr('fill', 'rgba(0, 0, 0, 0.5)')
      .attr('rx', 3)
      .attr('ry', 3);
    
    // The actual text labels
    const textLabels = nodeLabels.append('text')
      .attr('class', 'label-text')
      .text(d => {
        const labelText = d.labels ? (d.labels[0] || d.id) : d.id;
        return typeof labelText === 'string' ? labelText.substring(0, 30) : '';
      })
      .attr('font-size', '10px')
      .attr('fill', '#ffffff')
      .attr('dy', '0.35em')
      .attr('text-anchor', 'middle');
    
    // Size the background rectangles based on the text size
    nodeLabels.selectAll('.label-bg')
      .attr('width', function() {
        const textNode = d3.select(this.parentNode).select('text').node();
        if (!textNode) return 0;
        const length = textNode.getComputedTextLength();
        return isNaN(length) ? 0 : length + 6;
      })
      .attr('height', 16)
      .attr('x', function() {
        const textNode = d3.select(this.parentNode).select('text').node();
        if (!textNode) return 0;
        const length = textNode.getComputedTextLength();
        return isNaN(length) ? 0 : -length / 2 - 3;
      })
      .attr('y', -8);
    
    // Add drag behavior
    node.call(d3.drag()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
    );
    
    // Update positions on each simulation tick
    simulation.on('tick', () => {
      // Do NOT bound node positions - allow them to spread naturally
      // This prevents the flattening effect at the edges
      
      // Update link positions
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => {
          // Calculate endpoint to stop at node edge
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance === 0) return d.target.x; // Handle zero distance case
          const unitX = dx / distance;
          return d.target.x - unitX * nodeRadius;
        })
        .attr('y2', d => {
          // Calculate endpoint to stop at node edge
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance === 0) return d.target.y; // Handle zero distance case
          const unitY = dy / distance;
          return d.target.y - unitY * nodeRadius;
        });
      
      // Update node positions
      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y);
      
      // Update link label positions
      linkLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);
      
      // Update node label positions
      nodeLabels
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    // Add double-click to reset zoom
    svgElement.on('dblclick.zoom', () => {
      svgElement.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity);
    });
    
    // Handle window resize to update graph dimensions
    const handleResize = () => {
      const newWidth = window.innerWidth;
      const newHeight = window.innerHeight;
      
      svgElement
        .attr('width', newWidth)
        .attr('height', newHeight);
      
      simulation
        .force('center', d3.forceCenter(newWidth / 2, newHeight / 2))
        .alpha(0.3)
        .restart();
    };
    
    window.addEventListener('resize', handleResize);
    
    // Node hover effects for better interactivity
    node
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', nodeRadius * 1.5)
          .attr('fill', '#ff7f0e');
        
        // Highlight connected links
        link
          .style('stroke', l => (l.source.id === d.id || l.target.id === d.id) ? '#ff7f0e' : '#999')
          .style('stroke-opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.2)
          .style('stroke-width', l => (l.source.id === d.id || l.target.id === d.id) ? 2 : 1);
        
        // Always show label for hovered node, even if labels are off
        nodeLabels
          .style('opacity', n => {
            if (n.id === d.id) return 1;
            return showNodeLabels ? 0.3 : 0;
          });
        
        // Show relationship labels for connected links on hover
        if (!showRelationLabels) {
          linkLabels
            .style('opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0);
        }
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', nodeRadius)
          .attr('fill', '#1f77b4');
        
        // Reset link appearance
        link
          .style('stroke', '#999')
          .style('stroke-opacity', 0.6)
          .style('stroke-width', 1);
        
        // Reset node label appearance
        nodeLabels
          .style('opacity', showNodeLabels ? 1 : 0);
        
        // Reset relationship labels
        linkLabels
          .style('opacity', showRelationLabels ? 0.7 : 0);
      });
    
    // Initial zoom to show the entire graph - zoom out further
    setTimeout(() => {
      svgElement.call(zoom.transform, d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(0.2) // Zoom out more
        .translate(-width / 2, -height / 2));
    }, 500);
    
    // Run simulation longer to ensure better initial layout
    for (let i = 0; i < 50; i++) {
      simulation.tick();
    }
    
    console.log('Graph rendering complete');
    
    // Cleanup: Remove the event listener when component unmounts
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  };
  
  const toggleNodeLabels = () => {
    setShowNodeLabels(!showNodeLabels);
  };
  
  const toggleRelationLabels = () => {
    setShowRelationLabels(!showRelationLabels);
  };
  
  return (
    <div className="graph-container">
      <div className="header">
        <h1>Knowledge Graph Visualization</h1>
        {!loading && (
          <div className="stats">
            <span>{graphStats.nodes} Nodes</span>
            <span>{graphStats.edges} Relationships</span>
          </div>
        )}
      </div>
      
      {loading && <div className="loading">Loading graph data...</div>}
      
      <svg ref={svgRef} width={window.innerWidth} height={window.innerHeight}></svg>
      
      <div className="controls">
        <div className="toggles">
          <button 
            className={`toggle-btn ${showNodeLabels ? 'active' : ''}`}
            onClick={toggleNodeLabels}
          >
            {showNodeLabels ? 'Hide Node Labels' : 'Show Node Labels'}
          </button>
          <button 
            className={`toggle-btn ${showRelationLabels ? 'active' : ''}`}
            onClick={toggleRelationLabels}
          >
            {showRelationLabels ? 'Hide Relationship Labels' : 'Show Relationship Labels'}
          </button>
        </div>
        <div className="instructions">
          <p>Scroll to zoom • Double-click to reset view • Hover over nodes to see connections</p>
        </div>
      </div>
    </div>
  );
}

export default App;